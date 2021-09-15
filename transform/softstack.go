package transform

import (
	"errors"
	"fmt"
	"sort"

	"github.com/tinygo-org/tinygo/compiler/ircheck"
	"github.com/tinygo-org/tinygo/compiler/llvmutil"
	"tinygo.org/x/go-llvm"
)

func SoftStackify(mod llvm.Module) {
	llvm.VerifyModule(mod, llvm.AbortProcessAction)
	funcs := findBlockingCalls(mod)
	for _, fn := range funcs {
		lowerBlockingFunc(fn, mod)
	}

	println("pass done, transformed IR:")
	mod.Dump()
	println()

	llvm.VerifyModule(mod, llvm.AbortProcessAction)
	ircheck.Module(mod)
}

type blockingFunc struct {
	fn    llvm.Value
	calls []llvm.Value
}

var definitelyNotBlockingFuncs = map[string]struct{}{
	// This contains an indirect call which is used to launch the rewind operation.
	// As such, it can potentially confuse the findBlockingCalls pass.
	"(*internal/task.Task).Resume": {},

	// This does block, but must not be lowered (the pause operation is implemented manually).
	"internal/task.Pause": {},

	// Do not treat debug markers as suspending functions.
	// That can make things explode.
	"llvm.dbg.addr":    {},
	"llvm.dbg.declare": {},
	"llvm.dbg.value":   {},

	"llvm.lifetime.start.p0i8": {},
	"llvm.lifetime.end.p0i8":   {},

	"llvm.trap": {},
	"putchar":   {},
	"abort":     {},
}

// findBlockingCalls finds defined functions which may be async, and finds all potentially suspending calls within them.
func findBlockingCalls(mod llvm.Module) []blockingFunc {
	// List all *defined* functions in the order that they are defined within the module.
	// This is used to force the IR to be transformed in a consistent order.
	var funcs []blockingFunc
	for fn := mod.FirstFunction(); !fn.IsNil(); fn = llvm.NextFunction(fn) {
		if fn.IsDeclaration() {
			// This function is defined elsewhere, and therefore is not to be transformed here.
			continue
		}

		if _, ok := definitelyNotBlockingFuncs[fn.Name()]; ok {
			// This function definitely does not block.
			continue
		}

		funcs = append(funcs, blockingFunc{fn: fn})
	}
	fnMap := make(map[llvm.Value]*blockingFunc, len(funcs))
	for i := range funcs {
		fnMap[funcs[i].fn] = &funcs[i]
	}

	// Track a list of calls to functions which are not yet known to block.
	type maybeBlockingCall struct {
		inst   llvm.Value
		caller *blockingFunc
	}
	maybeCalls := make(map[llvm.Value][]maybeBlockingCall)

	// Scan the contents of every function for calls.
	// This is mainly necessary to handle indirect calls correctly.
	// TODO: handle "invoke".
	for i := range funcs {
		fn := &funcs[i]
		for bb := fn.fn.FirstBasicBlock(); !bb.IsNil(); bb = llvm.NextBasicBlock(bb) {
			for inst := bb.FirstInstruction(); !inst.IsNil(); inst = llvm.NextInstruction(inst) {
				if inst.IsACallInst().IsNil() {
					// This instruction is not a call.
					continue
				}

				callee := inst.CalledValue()
				switch {
				case !callee.IsConstant():
					// Indirect calls may or may not be able to block.
					// For safety, they must be treated as if they can block.
					fn.calls = append(fn.calls, inst)

				case callee.IsDeclaration():
					if _, ok := definitelyNotBlockingFuncs[callee.Name()]; ok {
						// This function definitely does not block.
						continue
					}

					// This function is defined elsewhere, and therefore it is not known to not block.
					fn.calls = append(fn.calls, inst)

				default:
					// This is a direct call to a function which may block.
					// Save it for later.
					maybeCalls[callee] = append(maybeCalls[callee], maybeBlockingCall{
						inst:   inst,
						caller: fn,
					})
				}
			}
		}
	}

	// Now that some functions have been marked as blocking, we can mark any known callers as blocking too.
	var workstack []maybeBlockingCall
	if pause := mod.NamedFunction("internal/task.Pause"); !pause.IsNil() {
		// Flush direct calls to Pause to the work stack.
		workstack = append(workstack, maybeCalls[pause]...)
		delete(maybeCalls, pause)
	}
	for _, fn := range funcs {
		if len(fn.calls) == 0 {
			// This function is not currently known to be async.
			continue
		}

		// Flush known calls to this function to the work stack.
		workstack = append(workstack, maybeCalls[fn.fn]...)
		delete(maybeCalls, fn.fn)
	}
	for len(workstack) > 0 {
		// Remove the top call of the work stack.
		call := workstack[len(workstack)-1]
		workstack = workstack[:len(workstack)-1]

		// Is this the first potentially blocking call by this caller?
		if len(call.caller.calls) == 0 {
			// Flush known calls to the caller to the work stack.
			workstack = append(workstack, maybeCalls[call.caller.fn]...)
			delete(maybeCalls, call.caller.fn)
		}

		// Add this call to the caller's list of blocking calls.
		call.caller.calls = append(call.caller.calls, call.inst)
	}

	// Isolate only the functions which contain potentially-blocking calls.
	blockingFuncs := funcs[:0]
	for _, fn := range funcs {
		if len(fn.calls) == 0 {
			// This function contains no potentially-blocking calls.
			continue
		}

		blockingFuncs = append(blockingFuncs, fn)
	}

	return blockingFuncs
}

func lowerBlockingFunc(fn blockingFunc, mod llvm.Module) {
	fmt.Println("processing", fn.fn.Name())
	ctx := mod.Context()

	// Find the windchain global variable.
	// This is used to handle unwinding and rewinding.
	windchain := mod.NamedGlobal("(internal/task).windchain")
	if windchain.IsNil() {
		// The windchain is not in this module.
		// Create a reference to it.
		windchain = llvm.AddGlobal(mod, llvm.PointerType(ctx.Int8Type(), 0), "(internal/task).windchain")
		windchain.SetLinkage(llvm.ExternalLinkage) // TODO: is this the right linkage?
	}
	windType := windchain.Type().ElementType()

	// Find the stack pointer global variable.
	// This is used to implement the software-stack.
	sp := mod.NamedGlobal("(internal/task).sp")
	if sp.IsNil() {
		// The stack pointer is not in this module.
		// Create a reference to it.
		sp = llvm.AddGlobal(mod, llvm.PointerType(ctx.Int8Type(), 0), "(internal/task).sp")
		sp.SetLinkage(llvm.ExternalLinkage) // TODO: is this the right linkage?
	}

	// Create a builder.
	builder := ctx.NewBuilder()
	defer builder.Dispose()

	// Obtain target data.
	target := llvm.NewTargetData(mod.DataLayout())
	defer target.Dispose()

	// Lower the stack allocations.
	lowerAllocas(ctx, mod, builder, target, fn.fn, sp)

	// Split the blocks such that each blocking call has its own block.
	// TODO: handle "invoke".
	type callState struct {
		name                  string
		call                  llvm.Value
		before, during, after llvm.BasicBlock
		rewind                llvm.BasicBlock
	}
	calls := make([]callState, len(fn.calls))
	for i, c := range fn.calls {
		// Save the block that the call is currently in.
		before := c.InstructionParent()
		name := before.AsValue().Name() + ".call"

		// Move instructions after the call into a new block.
		after := llvmutil.SplitBasicBlock(builder, c, before, name+".after")

		// Move the call into a new block after the original.
		callBlock := ctx.AddBasicBlock(fn.fn, name)
		callBlock.MoveAfter(before)
		callName := c.Name()
		c.RemoveFromParentAsInstruction()
		builder.SetInsertPointAtEnd(callBlock)
		builder.InsertWithName(c, callName)
		after.MoveAfter(callBlock)

		// Insert temporary branch instructions connecting them (for the PHI insertion pass to use for scanning).
		builder.CreateBr(after)
		builder.SetInsertPointAtEnd(before)
		builder.CreateBr(callBlock)

		calls[i] = callState{
			name:   name,
			call:   c,
			before: before,
			during: callBlock,
			after:  after,
		}
	}

	// Route the instruction saves.
	suspendBlocks := make([]llvm.BasicBlock, len(calls))
	for i := range suspendBlocks {
		suspendBlocks[i] = calls[i].during
	}
	routeInstSaves(builder, fn.fn, suspendBlocks...)

	// Split the entry block.
	entry := fn.fn.FirstBasicBlock()
	var normalEntry llvm.BasicBlock
	{
		// Find the instruction to split at.
		// This must be after all entry alloca instructions.
		splitInst := entry.FirstInstruction()
		for !splitInst.IsAAllocaInst().IsNil() {
			splitInst = llvm.NextInstruction(splitInst)
		}

		// Move the rest of the entry into a new block.
		normalEntry = splitBasicBlockAt(builder, splitInst, entry, "entry.normal")
		normalEntry.MoveAfter(entry)
	}

	// Accumulate saves.
	saves := make(map[llvm.BasicBlock][]llvm.Value)
	for i := range calls {
		b := calls[i].during
		var savePHIs []llvm.Value
		for i := b.FirstInstruction(); !i.IsAPHINode().IsNil(); i = llvm.NextInstruction(i) {
			savePHIs = append(savePHIs, i)
		}
		if len(savePHIs) == 0 {
			continue
		}

		saves[b] = savePHIs
	}

	// Create rewind blocks for every call.
	for i := range calls {
		var b llvm.BasicBlock
		if during := calls[i].during; len(saves[during]) == 0 {
			b = during
		} else {
			b = ctx.AddBasicBlock(fn.fn, calls[i].name+".rewind")
			b.MoveAfter(calls[i].before)
		}
		calls[i].rewind = b
	}

	// Create a main rewind path.
	var rewindBlock llvm.BasicBlock
	if len(calls) == 1 {
		rewindBlock = calls[0].rewind
	} else {
		rewindBlock = ctx.AddBasicBlock(fn.fn, "entry.rewind")
		rewindBlock.MoveAfter(entry)
	}
	builder.SetInsertPointAtEnd(entry)
	rewindPtr := builder.CreateLoad(windchain, "rewind.ptr")
	builder.CreateCondBr(
		builder.CreateICmp(llvm.IntNE, rewindPtr, llvm.ConstNull(rewindPtr.Type()), "rewind.check"),
		rewindBlock,
		normalEntry,
	)
	if len(calls) == 1 && rewindBlock == calls[0].during {
		// This is a super-special case: the function does not need to save *any state* (including control flow).
		// Insert an unwind check after the blocking call and then we are done.
		unwindBlock := ctx.AddBasicBlock(fn.fn, "unwind")
		unwindBlock.MoveAfter(rewindBlock)
		builder.SetInsertPointAtEnd(unwindBlock)
		if rtype := fn.fn.Type().ElementType().ReturnType(); rtype.TypeKind() == llvm.VoidTypeKind {
			builder.CreateRetVoid()
		} else {
			builder.CreateRet(llvm.Undef(rtype))
		}
		rewindBlock.LastInstruction().RemoveFromParentAsInstruction()
		builder.SetInsertPointAtEnd(rewindBlock)
		unwindPtr := builder.CreateLoad(windchain, "unwind.ptr")
		builder.CreateCondBr(
			builder.CreateICmp(llvm.IntNE, unwindPtr, llvm.ConstNull(unwindPtr.Type()), "unwind.check"),
			unwindBlock,
			calls[0].after,
		)
		return
	}
	builder.SetInsertPointAtEnd(rewindBlock)
	{
		// Move to the next element of the rewind chain.
		nextPtr := builder.CreateBitCast(rewindPtr, llvm.PointerType(windType, 0), "rewind.next.ptr")
		next := builder.CreateLoad(nextPtr, "rewind.next")
		builder.CreateStore(next, windchain)
	}
	var rewindIndexType llvm.Type
	if len(calls) != 1 {
		// Identify the correct rewind point and switch to it.
		//rewindIndexType = ctx.IntType(bits.Len(uint(len(calls) - 1)))
		rewindIndexType = ctx.Int32Type()
		rewindPackType := ctx.StructType([]llvm.Type{
			windType,
			rewindIndexType,
		}, false)
		pack := builder.CreateBitCast(rewindPtr, llvm.PointerType(rewindPackType, 0), "rewind.pack")
		idxPtr := builder.CreateStructGEP(pack, 1, "rewind.pack.idx")
		idx := builder.CreateLoad(idxPtr, "rewind.idx")
		unreachableBlock := ctx.AddBasicBlock(fn.fn, "rewind.unreachable")
		unreachableBlock.MoveAfter(rewindBlock)
		sw := builder.CreateSwitch(idx, unreachableBlock, len(calls))
		for i := range calls {
			sw.AddCase(llvm.ConstInt(rewindIndexType, uint64(i), false), calls[i].rewind)
		}
		builder.SetInsertPointAtEnd(unreachableBlock)
		builder.CreateUnreachable()
	} else if calls[0].rewind == calls[0].during {
		builder.CreateBr(calls[0].during)
	}

	// Fill in the rewind and unwind paths for each suspend point.
	for i, c := range calls {
		name := c.name

		// Place the saves in alignment-size order.
		saves := saves[c.during]
		sort.SliceStable(saves, func(i, j int) bool {
			itype := saves[i].Type()
			jtype := saves[j].Type()

			// Compare alignment.
			ialign := target.ABITypeAlignment(itype)
			jalign := target.ABITypeAlignment(jtype)
			switch {
			case ialign < jalign:
				return false
			case ialign > jalign:
				return true
			}

			// Compare size.
			isize := target.TypeAllocSize(itype)
			jsize := target.TypeAllocSize(jtype)
			return isize < jsize
		})

		// Create a "frame" type to hold all of the elements.
		var frameType llvm.Type
		{
			elemTypes := make([]llvm.Type, len(saves))
			for i, s := range saves {
				elemTypes[i] = s.Type()
			}

			frameType = ctx.StructType(elemTypes, false)
		}
		println("frame:", frameType.String())

		// Wrap the frame and add metadata.
		var foff int
		var superFrameType llvm.Type
		if rewindIndexType.IsNil() {
			foff = 1
			superFrameType = ctx.StructType([]llvm.Type{
				windType,
				frameType,
			}, false)
		} else {
			foff = 2
			superFrameType = ctx.StructType([]llvm.Type{
				windType,
				rewindIndexType,
				frameType,
			}, false)
		}

		if len(saves) > 0 {
			// Load all of the values from the frame.
			builder.SetInsertPointAtEnd(c.rewind)
			superFrame := builder.CreateBitCast(rewindPtr, llvm.PointerType(superFrameType, 0), name+".rewind.superframe")
			frame := builder.CreateStructGEP(superFrame, foff, name+".rewind.frame")
			reloads := make([]llvm.Value, len(saves))
			for i := range saves {
				srcName := saves[i].Name()
				ptr := builder.CreateStructGEP(frame, i, srcName+".reload.ptr")
				reloads[i] = builder.CreateLoad(ptr, srcName+".reload")
			}

			// Merge back into normal flow.
			builder.CreateBr(c.during)
			for i, s := range saves {
				s.AddIncoming(
					[]llvm.Value{reloads[i]},
					[]llvm.BasicBlock{c.rewind},
				)
			}
		}

		// Create an unwind path.
		unwindBlock := ctx.AddBasicBlock(fn.fn, name+".unwind")
		unwindBlock.MoveAfter(c.during)
		c.during.LastInstruction().RemoveFromParentAsInstruction()
		builder.SetInsertPointAtEnd(c.during)
		unwindPtr := builder.CreateLoad(windchain, name+".unwind.ptr")
		builder.CreateCondBr(
			builder.CreateICmp(llvm.IntNE, unwindPtr, llvm.ConstNull(windType), name+".unwind.check"),
			unwindBlock,
			c.after,
		)

		// Allocate a superframe on the stack.
		builder.SetInsertPointAtEnd(unwindBlock)
		unwindFrame := createSoftStackAlloc(ctx, builder, target, sp, superFrameType, llvm.ConstInt(ctx.Int1Type(), 1, false), name+".unwind.frame")

		// Save the previous unwind pointer.
		prevPtr := builder.CreateStructGEP(unwindFrame, 0, name+".unwind.frame.prev")
		builder.CreateStore(unwindPtr, prevPtr)

		if !rewindIndexType.IsNil() {
			// Save the rewind index.
			idxPtr := builder.CreateStructGEP(unwindFrame, 1, name+".unwind.frame.idx")
			idx := llvm.ConstInt(rewindIndexType, uint64(i), false)
			builder.CreateStore(idx, idxPtr)
		}

		if len(saves) > 0 {
			// Fill the frame.
			frame := builder.CreateStructGEP(unwindFrame, foff, name+".unwind.frame.frame")
			for i, save := range saves {
				ptr := builder.CreateStructGEP(frame, i, save.Name()+".ptr")
				builder.CreateStore(save, ptr)
			}
		}

		// Push the frame onto the unwind chain.
		unwindFrameCast := builder.CreateBitCast(unwindFrame, windType, name+".unwind.frame.cast")
		builder.CreateStore(unwindFrameCast, windchain)

		// Exit the function.
		if rtype := fn.fn.Type().ElementType().ReturnType(); rtype.TypeKind() == llvm.VoidTypeKind {
			builder.CreateRetVoid()
		} else {
			builder.CreateRet(llvm.Undef(rtype))
		}
	}

	llvm.VerifyFunction(fn.fn, llvm.AbortProcessAction)
}

func routeInstSaves(builder llvm.Builder, fn llvm.Value, suspends ...llvm.BasicBlock) map[llvm.BasicBlock][]llvm.Value {
	fn.Dump()
	println()
	llvm.VerifyFunction(fn, llvm.AbortProcessAction)
	if len(suspends) == 0 {
		panic(errors.New("no suspends"))
	}

	// Build a control-flow graph of the function.
	cfg := buildCFG(fn)

	// Build a dominator tree.
	dom := cfg.analyzeDominance()

	// Prepare the suspend points.
	saves := make(map[llvm.BasicBlock][]llvm.Value, len(suspends))
	for _, sus := range suspends {
		saves[sus] = nil
	}

	// Gather all of the instructions in the function that produce a value.
	// This must be done before saves are routed in order to avoid processing created PHI nodes.
	var insts []llvm.Value
	for _, b := range cfg.blocks {
		for i := b.FirstInstruction(); !i.IsNil(); i = llvm.NextInstruction(i) {
			if i.Type().TypeKind() == llvm.VoidTypeKind {
				continue
			}

			insts = append(insts, i)
		}
	}

	// Route each instruction through appropriate suspend points.
	// This is done backwards in order to put PHI nodes in a sane order.
	pass := instSavesPass{
		builder:  builder,
		cfg:      cfg,
		dom:      dom,
		suspends: saves,
	}
	for i := len(insts) - 1; i >= 0; i-- {
		pass.applyInstSaves(insts[i])
	}

	fn.Dump()
	println()

	return pass.suspends
}

type instSavesPass struct {
	builder llvm.Builder

	cfg cfg
	dom dominanceAnalysis

	suspends map[llvm.BasicBlock][]llvm.Value
}

func (p *instSavesPass) applyInstSaves(def llvm.Value) {
	print("apply inst saves for:")
	def.Dump()
	println()

	// Find all uses of the instruction, except uses within the same block.
	type userInfo struct {
		user  llvm.Value
		block llvm.BasicBlock
	}
	var uses []userInfo
	defBlock := def.InstructionParent()
	for use := def.FirstUse(); !use.IsNil(); use = use.NextUse() {
		user := use.User()

		if !user.IsAPHINode().IsNil() {
			// PHI nodes must be handled specially.
			// The source for a PHI node is effectively the corresponding predecessor block.
			n := user.IncomingCount()
			for i := 0; i < n; i++ {
				if user.IncomingValue(i) != def {
					continue
				}

				srcBlock := user.IncomingBlock(i)
				if srcBlock == defBlock {
					// The block loops directly to itself, and retains the value.
					// This path does not require transformation.
					continue
				}

				uses = append(uses, userInfo{
					user:  user,
					block: srcBlock,
				})
			}
			continue
		}

		block := user.InstructionParent()
		if block == defBlock {
			// This is within the same block, so no transformation is necessary.
			continue
		}
		uses = append(uses, userInfo{
			user:  user,
			block: block,
		})
	}
	if len(uses) == 0 {
		// The instruction is only used within one block, so no saving is necessary.
		println("\tskipping instruction: no uses")
		return
	}

	// Find all blocks through which the value must pass.
	var blocks []llvm.BasicBlock
	liveSet := make(map[llvm.BasicBlock]struct{})
	{
		for _, u := range uses {
			if _, ok := liveSet[u.block]; ok {
				continue
			}

			blocks = append(blocks, u.block)
			liveSet[u.block] = struct{}{}
		}
		for i := 0; i < len(blocks); i++ {
			b := blocks[i]
			for _, pred := range p.cfg.predecessors[b] {
				if pred == defBlock {
					continue
				}

				if _, ok := liveSet[pred]; ok {
					continue
				}

				blocks = append(blocks, pred)
				liveSet[pred] = struct{}{}
			}
		}
	}
	println("\tlive blocks:")
	for _, b := range blocks {
		println("\t\t" + b.AsValue().Name())
	}

	// Find all save points through which the value passes.
	var saveBlocks []llvm.BasicBlock
	for _, b := range blocks {
		if _, ok := p.suspends[b]; ok {
			saveBlocks = append(saveBlocks, b)
		}
	}
	if len(saveBlocks) == 0 {
		// This value does not pass through any save points.
		// No saves are necessary.
		println("\tskipping instruction: no save blocks")
		return
	}

	// Insert a PHI node at each save point.
	savePHIs := make([]llvm.Value, len(saveBlocks))
	t := def.Type()
	name := def.Name()
	for i, b := range saveBlocks {
		p.builder.SetInsertPointBefore(b.FirstInstruction())
		savePHIs[i] = p.builder.CreatePHI(t, fmt.Sprintf("%s.save.%d", name, i))
		println("\tinserting save PHI at", b.AsValue().Name())
	}

	// Compute the join points between the definition and all save points.
	joins := p.dom.joins(append([]llvm.BasicBlock{defBlock}, saveBlocks...), liveSet)

	// Insert a PHI node at each join point.
	joinPHIs := make([]llvm.Value, len(joins))
	for i, j := range joins {
		p.builder.SetInsertPointBefore(j.FirstInstruction())
		joinPHIs[i] = p.builder.CreatePHI(t, fmt.Sprintf("%s.join.%d", name, i))
		println("\tinserting join PHI at", j.AsValue().Name())
	}

	// Create a mapping of all blocks to their corresponding source instruction.
	srcMap := make(map[llvm.BasicBlock]llvm.Value, 1+len(saveBlocks)+len(joins))
	srcMap[defBlock] = def
	for i, b := range saveBlocks {
		srcMap[b] = savePHIs[i]
	}
	for i, b := range joins {
		srcMap[b] = joinPHIs[i]
	}
	for _, b := range blocks {
		var stack []llvm.BasicBlock
		var src llvm.Value
		for {
			if val, ok := srcMap[b]; ok {
				src = val
				break
			}

			stack = append(stack, b)
			b = p.dom.idoms[b]
		}
		for _, b := range stack {
			srcMap[b] = src
		}
	}

	// Resolve all join references by walking the dominator tree until something is found in srcMap.
	for i, block := range joins {
		preds := p.cfg.predecessors[block]
		predVals := make([]llvm.Value, len(preds))
		for i, b := range preds {
			predVals[i] = srcMap[b]
		}

		joinPHIs[i].AddIncoming(predVals, preds)
	}

	// Fill in all of the uses.
	for _, use := range uses {
		// Resolve the source.
		src := srcMap[use.block]
		if src == def {
			// This use does not require transformation.
			continue
		}

		// Update operands to refer to the new source.
		switch use.user.InstructionOpcode() {
		case llvm.PHI:
			// Update only the correct incoming path.
			n := use.user.IncomingCount()
			for i := 0; i < n; i++ {
				if use.user.IncomingBlock(i) == use.block {
					use.user.SetOperand(i, src)
				}
			}

		default:
			// Scan all operands and replace any containing the definition value.
			n := use.user.OperandsCount()
			for i := 0; i < n; i++ {
				if use.user.Operand(i) == def {
					use.user.SetOperand(i, src)
				}
			}
		}
	}

	// Add save-point PHIs to the suspend saves lists.
	for i, b := range saveBlocks {
		// This requires the source for the previous block, not the PHI in this block.
		src := srcMap[p.dom.idoms[b]]
		phi := savePHIs[i]
		phi.AddIncoming(
			[]llvm.Value{src},
			[]llvm.BasicBlock{p.cfg.predecessors[phi.InstructionParent()][0]},
		)
		print("\tsaved at ", phi.InstructionParent().AsValue().Name(), " with source value")
		src.Dump()
		println(" from", src.InstructionParent().AsValue().Name())
		p.suspends[b] = append(p.suspends[b], phi)
	}
}

type dominanceAnalysis struct {
	idoms map[llvm.BasicBlock]llvm.BasicBlock
	df    map[llvm.BasicBlock][]llvm.BasicBlock
}

func (da *dominanceAnalysis) joins(sources []llvm.BasicBlock, lifetime map[llvm.BasicBlock]struct{}) []llvm.BasicBlock {
	// Start by unioning the source dominance frontiers.
	var joins []llvm.BasicBlock
	dedup := make(map[llvm.BasicBlock]struct{})
	for _, src := range sources {
		for _, b := range da.df[src] {
			if _, ok := lifetime[b]; !ok {
				continue
			}

			if _, ok := dedup[b]; ok {
				continue
			}

			joins = append(joins, b)
			dedup[b] = struct{}{}
		}
	}

	// Extend the set to include the dominance frontiers of every node in the set.
	for i := 0; i < len(joins); i++ {
		for _, b := range da.df[joins[i]] {
			if _, ok := lifetime[b]; !ok {
				continue
			}

			if _, ok := dedup[b]; ok {
				continue
			}

			joins = append(joins, b)
			dedup[b] = struct{}{}
		}
	}

	return joins
}

func (c cfg) analyzeDominance() dominanceAnalysis {
	// This uses https://www.cs.rice.edu/~keith/EMBED/dom.pdf to compute dominance frontiers.

	// Place the blocks in postorder.
	blocks := c.postorder()

	// Assign an index to each block.
	blockIdx := make(map[llvm.BasicBlock]int, len(blocks))
	for i, block := range blocks {
		blockIdx[block] = i
	}

	// Reorganize the predecessor graph by index.
	predecessors := make([][]int, len(blocks))
	for dst, srcs := range c.predecessors {
		srcIdxs := make([]int, len(srcs))
		for i, src := range srcs {
			srcIdxs[i] = blockIdx[src]
		}
		sort.Ints(srcIdxs)
		predecessors[blockIdx[dst]] = srcIdxs
	}

	// Iteratively calculate immediate dominators.
	idoms := make([]int, len(blocks))
	for i := range idoms {
		idoms[i] = -1
	}
	idoms[len(idoms)-1] = len(idoms) - 1
	for {
		var changed bool
		for b := len(blocks) - 2; b >= 0; b-- {
			preds := predecessors[b]
			newIdom := -1
			if len(preds) == 0 {
				panic(fmt.Errorf("missing preds in block %q", blocks[b].AsValue().Name()))
			}
			for _, p := range preds {
				if p == -1 {
					panic(errors.New("WAT"))
				}
				if idoms[p] != -1 {
					newIdom = p
				}
			}
			if newIdom == -1 {
				panic(fmt.Errorf("no processed preds for block %q", blocks[b].AsValue().Name()))
			}
			for _, p := range preds {
				if idoms[p] == -1 {
					continue
				}

				x, y := p, newIdom
				for x != y {
					for x < y {
						x = idoms[x]
					}
					for y < x {
						y = idoms[y]
					}
				}
				newIdom = x
			}
			if idoms[b] != newIdom {
				idoms[b] = newIdom
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	// Calculate dominance frontiers.
	df := make([][]int, len(blocks))
	for b := range blocks {
		preds := predecessors[b]
		if len(preds) < 2 {
			continue
		}

		for _, p := range preds {
			runner := p
			for runner != idoms[b] {
				df[runner] = append(df[runner], b)
				runner = idoms[runner]
			}
		}
	}

	// Map the results back to basic blocks for convenience.
	idomsBB := make(map[llvm.BasicBlock]llvm.BasicBlock, len(idoms))
	for v, w := range idoms {
		idomsBB[blocks[v]] = blocks[w]
	}
	dfBB := make(map[llvm.BasicBlock][]llvm.BasicBlock, len(df))
	for v, set := range df {
		setBB := make([]llvm.BasicBlock, len(set))
		for i, w := range set {
			setBB[i] = blocks[w]
		}
		dfBB[blocks[v]] = setBB
	}

	return dominanceAnalysis{
		idoms: idomsBB,
		df:    dfBB,
	}
}

type cfg struct {
	blocks       []llvm.BasicBlock
	successors   map[llvm.BasicBlock][]llvm.BasicBlock
	predecessors map[llvm.BasicBlock][]llvm.BasicBlock
}

func (c *cfg) postorder() []llvm.BasicBlock {
	walker := postorderWalker{
		visited:    make(map[llvm.BasicBlock]struct{}, len(c.blocks)),
		successors: c.successors,
	}

	walker.walk(c.blocks[0])

	return walker.list
}

type postorderWalker struct {
	list       []llvm.BasicBlock
	visited    map[llvm.BasicBlock]struct{}
	successors map[llvm.BasicBlock][]llvm.BasicBlock
}

func (w *postorderWalker) walk(b llvm.BasicBlock) {
	w.visited[b] = struct{}{}
	for _, s := range w.successors[b] {
		if _, ok := w.visited[s]; ok {
			continue
		}

		w.walk(s)
	}

	w.list = append(w.list, b)
}

func buildCFG(fn llvm.Value) cfg {
	blocks := fn.BasicBlocks()

	successors := make(map[llvm.BasicBlock][]llvm.BasicBlock, len(blocks))
	predecessors := make(map[llvm.BasicBlock][]llvm.BasicBlock, len(blocks))
	for _, bb := range blocks {
		term := bb.LastInstruction()
		var dsts []llvm.BasicBlock
		switch term.InstructionOpcode() {
		case llvm.Ret, llvm.Unreachable:
			// These terminators lead out of the function.
			continue

		case llvm.Br:
			// The indices here are negative.
			/*
				dsts = []llvm.BasicBlock{
					term.Operand(-1).AsBasicBlock(),
				}
				if term.OperandsCount() == 3 {
					dsts = append(dsts, term.Operand(-2).AsBasicBlock())
				}*/
			switch term.OperandsCount() {
			case 1:
				dsts = []llvm.BasicBlock{
					term.Operand(0).AsBasicBlock(),
				}
			case 3:
				dsts = []llvm.BasicBlock{
					term.Operand(1).AsBasicBlock(),
					term.Operand(2).AsBasicBlock(),
				}
			}

		case llvm.Switch:
			// Destination blocks are in the odd operands.
			nops := term.OperandsCount()
			dsts = make([]llvm.BasicBlock, nops/2)
			for i := range dsts {
				dsts[i] = term.Operand(2*i + 1).AsBasicBlock()
			}

		case llvm.IndirectBr:
			// The list of destinations is from 1 and up.
			nops := term.OperandsCount()
			dsts = make([]llvm.BasicBlock, nops-1)
			for i := range dsts {
				dsts[i] = term.Operand(i + 1).AsBasicBlock()
			}

		case llvm.Invoke:
			// The normal destination is -3 and the exceptional destination is -2.\
			nops := term.OperandsCount()
			dsts = []llvm.BasicBlock{
				term.Operand(nops - 3).AsBasicBlock(),
				term.Operand(nops - 2).AsBasicBlock(),
			}

		default:
			term.Dump()
			panic(errors.New("unsupported terminator"))
		}
		{
			dedup := make(map[llvm.BasicBlock]struct{}, len(dsts))
			old := dsts
			dsts = dsts[:0]
			for _, dst := range old {
				if _, ok := dedup[dst]; ok {
					continue
				}

				dsts = append(dsts, dst)
			}
		}

		successors[bb] = dsts
		for _, dst := range dsts {
			predecessors[dst] = append(predecessors[dst], bb)
		}
	}

	return cfg{
		blocks:       blocks,
		successors:   successors,
		predecessors: predecessors,
	}
}

func lowerAllocas(ctx llvm.Context, mod llvm.Module, builder llvm.Builder, target llvm.TargetData, fn llvm.Value, sp llvm.Value) {
	fn.Dump()
	println()
	defer llvm.VerifyFunction(fn, llvm.AbortProcessAction)
	uintptrType := ctx.IntType(int(target.TypeSizeInBits(llvm.PointerType(ctx.Int8Type(), 0))))

	// Find all stack allocations.
	var frameAllocas, dynamicAllocas []llvm.Value
	entry := fn.FirstBasicBlock()
	for bb := entry; !bb.IsNil(); bb = llvm.NextBasicBlock(bb) {
		for inst := bb.FirstInstruction(); !inst.IsNil(); inst = llvm.NextInstruction(inst) {
			switch {
			case inst.IsAAllocaInst().IsNil():
				// This is not an alloca.

			case inst.InstructionParent() == entry && inst.Operand(0).IsConstant():
				// This is a fixed-sized stack allocation in the entry block.
				// It can be placed into a static frame.
				frameAllocas = append(frameAllocas, inst)

			default:
				// This allocation must be executed dynamically.
				// This shouldn't show up in Go code, but may appear in fancy C code.
				dynamicAllocas = append(dynamicAllocas, inst)
			}
		}
	}
	if len(frameAllocas) == 0 && len(dynamicAllocas) == 0 {
		// There are no allocas to lower.
		return
	}

	// Handle the extremely weird but technically valid case where there is an alloca instruction in the entry block, but not at the start.
	builder.SetInsertPointBefore(entry.FirstInstruction())
	for _, a := range frameAllocas {
		prev := llvm.PrevInstruction(a)
		if prev.IsNil() {
			// This is the first instruction, so it can be left as-is.
			continue
		}

		if prev.IsAAllocaInst().IsNil() {
			// The previous instruction is not an alloca instruction.
			// Move this to the start of the block.
			a.RemoveFromParentAsInstruction()
			builder.Insert(a)
		}
	}

	// Reorder frame allocas by alignment (and then size).
	sort.SliceStable(frameAllocas, func(i, j int) bool {
		itype := frameAllocas[i].Type().ElementType()
		jtype := frameAllocas[j].Type().ElementType()

		// Compare alignment.
		ialign := target.ABITypeAlignment(itype)
		jalign := target.ABITypeAlignment(jtype)
		switch {
		case ialign < jalign:
			return false
		case ialign > jalign:
			return true
		}

		// Compare size.
		isize := target.TypeAllocSize(itype)
		jsize := target.TypeAllocSize(jtype)
		return isize < jsize
	})

	// Construct the stack frame type.
	var frameType llvm.Type
	{
		// TODO: merge allocas, and potentially leave some of them on the hard stack.
		frameElems := make([]llvm.Type, len(frameAllocas))
		for i, alloca := range frameAllocas {
			// Identify the storage type for this alloca.
			t := alloca.Type().ElementType()
			len := alloca.Operand(0)
			if len := len.ZExtValue(); len != 1 {
				// This is an array alloca.
				// Store as an array.
				t = llvm.ArrayType(t, int(len))
			}

			frameElems[i] = t
		}

		// Create a struct to hold the elements.
		// The use of "packed" may seem strange, but it works because of the previous sort.
		frameType = ctx.StructType(frameElems, true)
	}

	// Split the entry block.
	var start llvm.BasicBlock
	{
		// Find the instruction to split at.
		// This must be after all entry alloca instructions.
		splitInst := entry.FirstInstruction()
		for !splitInst.IsAAllocaInst().IsNil() {
			splitInst = llvm.NextInstruction(splitInst)
		}

		// Move the rest of the entry into a new block.
		start = splitBasicBlockAt(builder, splitInst, entry, "entry.entry")
	}

	// Create a block to place the frame on the system stack when running outside of a goroutine.
	rawEntry := ctx.AddBasicBlock(fn, "entry.raw")
	rawEntry.MoveAfter(entry)
	builder.SetInsertPointAtEnd(rawEntry)
	rawFrame, _, _ := llvmutil.CreateTemporaryAlloca(builder, mod, frameType, "frame.raw")
	builder.CreateBr(start)

	// Create a block to place the frame on the soft stack when running inside a goroutine.
	softEntry := ctx.AddBasicBlock(fn, "entry.soft")
	softEntry.MoveAfter(entry)
	builder.SetInsertPointAtEnd(softEntry)
	softFrame := createSoftStackAlloc(ctx, builder, target, sp, frameType, rawFrame.Operand(0), "frame.soft")
	builder.CreateBr(start)

	// Detect whether the function is running on a goroutine and dispatch to the appropriate setup.
	builder.SetInsertPointAtEnd(entry)
	entrySP := builder.CreateLoad(sp, "entry.sp")
	onSystemStack := builder.CreateICmp(llvm.IntEQ, entrySP, llvm.ConstNull(entrySP.Type()), "entry.onSystemStack")
	builder.CreateCondBr(
		onSystemStack,
		rawEntry,
		softEntry,
	)

	// Insert the frame pointer as a PHI node into the start of the post-setup entry block.
	builder.SetInsertPointBefore(start.FirstInstruction())
	framePtr := builder.CreatePHI(llvm.PointerType(frameType, 0), "frame.ptr")
	framePtr.AddIncoming(
		[]llvm.Value{rawFrame, softFrame},
		[]llvm.BasicBlock{rawEntry, softEntry},
	)

	// Redirect frame alloca uses to the stack frame.
	// A seperate GEP is placed at each use in order to avoid saving interior pointers.
	// This may be cleaned up later by the optimizer.
	println("lowering frame allocas")
	framePtr.Dump()
	println()
	for i, a := range frameAllocas {
		print("\t")
		a.Dump()
		println()
		gepIndices := []llvm.Value{
			llvm.ConstInt(uintptrType, 0, false),
			llvm.ConstInt(ctx.Int32Type(), uint64(i), false),
		}
		if a.Operand(0).ZExtValue() != 1 {
			gepIndices = append(gepIndices, llvm.ConstInt(uintptrType, 0, false))
		}
		println("\t\tgep indices:")
		for _, i := range gepIndices {
			print("\t\t\t")
			i.Dump()
			println()
		}
		for use := a.FirstUse(); !use.IsNil(); use = a.FirstUse() {
			print("\t\tlowering use:")
			user := use.User()
			user.Dump()
			println()
			builder.SetInsertPointBefore(user)
			ptr := builder.CreateInBoundsGEP(framePtr, gepIndices, a.Name()+".ptr")
			numOps := user.OperandsCount()
			for i := 0; i < numOps; i++ {
				print("\t\t\toperand ", i, " ")
				user.Operand(i).Dump()
				if user.Operand(i) == a {
					user.SetOperand(i, ptr)
					print(" replaced by ")
					ptr.Dump()
				}
				println()
			}
		}

		if !a.FirstUse().IsNil() {
			println("\t\tremaining uses:")
			for use := a.FirstUse(); !use.IsNil(); use = use.NextUse() {
				print("\t\t\t")
				use.User().Dump()
				println()
			}
			panic(errors.New("remaining uses of alloca"))
		}

		a.RemoveFromParentAsInstruction()
	}

	// Lower dynamic allocas.
	for _, alloca := range dynamicAllocas {
		t := alloca.Type().ElementType()
		name := alloca.Name()
		alloca.SetName(name + ".raw")

		// Move the instructions after the alloca into a new block.
		beforeAlloc := alloca.InstructionParent()
		postAlloc := llvmutil.SplitBasicBlock(builder, alloca, beforeAlloc, beforeAlloc.AsValue().Name()+".continued")

		// Create a PHI node at the start of the post-alloc block to merge the alloca sources.
		builder.SetInsertPointBefore(postAlloc.FirstInstruction())
		newAlloca := builder.CreatePHI(llvm.PointerType(t, 0), name)
		alloca.ReplaceAllUsesWith(newAlloca)

		// Create a block allocating on the system stack.
		rawAllocaBlock := ctx.AddBasicBlock(fn, name+".raw.block")
		rawAllocaBlock.MoveAfter(beforeAlloc)
		builder.SetInsertPointAtEnd(rawAllocaBlock)
		alloca.RemoveFromParentAsInstruction()
		builder.Insert(alloca)
		builder.CreateBr(postAlloc)

		// Create a block allocating on the goroutine stack.
		softAllocaBlock := ctx.AddBasicBlock(fn, name+".soft.block")
		softAllocaBlock.MoveAfter(beforeAlloc)
		builder.SetInsertPointAtEnd(softAllocaBlock)
		softAlloca := createSoftStackAlloc(ctx, builder, target, sp, t, alloca.Operand(0), name+".soft")
		builder.CreateBr(postAlloc)

		// Merge the alloca sources into the PHI node.
		newAlloca.AddIncoming(
			[]llvm.Value{alloca, softAlloca},
			[]llvm.BasicBlock{rawAllocaBlock, softAllocaBlock},
		)

		// Select the appropriate path for the alloca.
		builder.SetInsertPointAtEnd(beforeAlloc)
		builder.CreateCondBr(onSystemStack, rawAllocaBlock, softAllocaBlock)
	}

	// Restore the stack pointer at each return.
	for bb := fn.FirstBasicBlock(); !bb.IsNil(); bb = llvm.NextBasicBlock(bb) {
		if term := bb.LastInstruction(); !term.IsAReturnInst().IsNil() {
			builder.SetInsertPointBefore(term)
			builder.CreateStore(entrySP, sp)
		}
	}
}

func createSoftStackAlloc(ctx llvm.Context, builder llvm.Builder, td llvm.TargetData, sp llvm.Value, t llvm.Type, count llvm.Value, name string) llvm.Value {
	// Calculate type allocation constraints.
	size := td.TypeAllocSize(t)
	align := td.ABITypeAlignment(t)

	// Reserve space on the stack.
	oldSP := builder.CreateLoad(sp, name+".sp")
	uintptrTypeWidth := int(td.TypeSizeInBits(oldSP.Type()))
	uintptrType := ctx.IntType(uintptrTypeWidth)
	spInt := builder.CreatePtrToInt(oldSP, uintptrType, name+".sp.int")
	countTypeWidth := count.Type().IntTypeWidth()
	switch {
	case countTypeWidth < uintptrTypeWidth:
		count = builder.CreateZExt(count, uintptrType, name+".n")
	case countTypeWidth > uintptrTypeWidth:
		count = builder.CreateTrunc(count, uintptrType, name+".n")
	}
	allocSize := builder.CreateMul(
		llvm.ConstInt(uintptrType, size, false),
		count,
		name+".size",
	)
	spSub := builder.CreateSub(spInt, allocSize, name+".sp.sub")
	spAlign := builder.CreateAnd(spSub, llvm.ConstNot(llvm.ConstInt(uintptrType, uint64(align-1), false)), name+".sp.align")
	spNew := builder.CreateIntToPtr(spAlign, oldSP.Type(), name+".sp.new")
	builder.CreateStore(spNew, sp)
	alloc := builder.CreateBitCast(spNew, llvm.PointerType(t, 0), name)

	return alloc
}

// TODO: dedup
func splitBasicBlockAt(builder llvm.Builder, atInst llvm.Value, insertAfter llvm.BasicBlock, name string) llvm.BasicBlock {
	oldBlock := atInst.InstructionParent()
	newBlock := atInst.Type().Context().InsertBasicBlock(insertAfter, name)
	newBlock.MoveAfter(insertAfter)   // TODO: why is this necessary??????
	var nextInstructions []llvm.Value // values to move

	// Collect to-be-moved instructions.
	for inst := atInst; !inst.IsNil(); inst = llvm.NextInstruction(inst) {
		nextInstructions = append(nextInstructions, inst)
	}

	// Move instructions.
	builder.SetInsertPointAtEnd(newBlock)
	for _, inst := range nextInstructions {
		instName := inst.Name()
		inst.RemoveFromParentAsInstruction()
		builder.InsertWithName(inst, instName)
	}

	// Find PHI nodes to update.
	var phiNodes []llvm.Value // PHI nodes to update
	for bb := insertAfter.Parent().FirstBasicBlock(); !bb.IsNil(); bb = llvm.NextBasicBlock(bb) {
		for inst := bb.FirstInstruction(); !inst.IsNil(); inst = llvm.NextInstruction(inst) {
			if inst.IsAPHINode().IsNil() {
				continue
			}
			needsUpdate := false
			incomingCount := inst.IncomingCount()
			for i := 0; i < incomingCount; i++ {
				if inst.IncomingBlock(i) == oldBlock {
					needsUpdate = true
					break
				}
			}
			if !needsUpdate {
				// PHI node has no incoming edge from the old block.
				continue
			}
			phiNodes = append(phiNodes, inst)
		}
	}

	// Update PHI nodes.
	for _, phi := range phiNodes {
		builder.SetInsertPointBefore(phi)
		newPhi := builder.CreatePHI(phi.Type(), "")
		incomingCount := phi.IncomingCount()
		incomingVals := make([]llvm.Value, incomingCount)
		incomingBlocks := make([]llvm.BasicBlock, incomingCount)
		for i := 0; i < incomingCount; i++ {
			value := phi.IncomingValue(i)
			block := phi.IncomingBlock(i)
			if block == oldBlock {
				block = newBlock
			}
			incomingVals[i] = value
			incomingBlocks[i] = block
		}
		name := phi.Name()
		newPhi.AddIncoming(incomingVals, incomingBlocks)
		phi.ReplaceAllUsesWith(newPhi)
		phi.EraseFromParentAsInstruction()
		newPhi.SetName(name)
	}

	return newBlock
}
