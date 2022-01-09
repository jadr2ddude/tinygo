//go:build gc.precise
// +build gc.precise

package runtime

import "unsafe"

// Set gcDebug to true to print debug information.
const (
	gcDebug   = false   // print debug info
	gcAsserts = gcDebug // perform sanity checks
)

const (
	ptrSize       = unsafe.Sizeof(unsafe.Pointer(nil))
	wordsPerBlock = 4 // number of pointers in an allocated block
	bytesPerBlock = wordsPerBlock * ptrSize
)

var (
	metadataStart unsafe.Pointer // pointer to the start of the heap metadata
	nextAlloc     gcBlock        // the next block that should be tried by the allocator
	endBlock      gcBlock        // the block just past the end of the available space
)

type gcBlockData [bytesPerBlock]byte

// gcBlock is a memory block index.
type gcBlock uintptr

// blockFromAddr returns a block given an address somewhere in the heap (which
// might not be heap-aligned).
func blockFromAddr(addr uintptr) gcBlock {
	if gcAsserts && (addr < heapStart || addr >= uintptr(metadataStart)) {
		runtimePanic("gc: trying to get block from invalid address")
	}
	return gcBlock((addr - heapStart) / bytesPerBlock)
}

func (b gcBlock) meta() *gcBlockMeta {
	return unsafe.Pointer(uintptr(metadataStart) + uintptr(b))
}

func (b gcBlock) data() *gcBlockData {
	return (*gcBlockData)(unsafe.Pointer(uintptr(heapStart) + bytesPerBlock*uintptr(b)))
}

func (b gcBlock) end() gcBlock {
	b++
	for b < endBlock && b.meta().state() == blockStateTail {
		b++
	}

	return b - 1
}

// gcBlockMeta holds metadata for a single memory block.
// The zero value represents an empty block.
// The lower 3 bits are used to store the gcBlockState.
// The fourth bit is used to implement the scan tree.
// See gc_precise_avr.go and gc_precise_other.go for the representation of the upper 4 bits.
type gcBlockMeta uint8

const (
	gcBlockStateMask    gcBlockMeta = 0b0111
	gcBlockChildPending gcBlockMeta = 0b1000
	gcBlockExtraMask    gcBlockMeta = 0b11110000
)

// state returns the gcBlockState of the block.
func (d gcBlockMeta) state() gcBlockState {
	return gcBlockState(d & gcBlockStateMask)
}

// setState updates the gcBlockState of the block.
func (d *gcBlockMeta) setState(state gcBlockState) {
	*d = (*d &^ gcBlockStateMask) | gcBlockMeta(state)
}

// maybeHasChildPending checks if there may be a scan-pending allocation underneath this block.
func (d gcBlockMeta) maybeHasChildPending() bool {
	return d&gcBlockTreePending != 0
}

// setChildPending flags the block as having a scan-pending allocation underneath.
func (d *gcBlockMeta) setChildPending() {
	*d |= gcBlockChildPending
}

// setChildPending flags the block as lacking a scan-pending allocation underneath.
func (d *gcBlockMeta) clearChildPending() {
	*d &^= gcBlockChildPending
}

func (d *gcBlockMeta) setExtra(extra uint8) uint8 {
	d = (d &^ gcBlockExtraMask) | (extra << 4)
}

func (d gcBlockMeta) extra() uint8 {
	return uint8(d >> 4)
}

// gcBlockState is the state of a single memory block.
type gcBlockState uint8

const (
	// blockStateEmpty is the state of a block which is not in use.
	blockStateEmpty gcBlockState = iota

	// blockStateHead is the state of a block which starts an allocation.
	// When a reference to the allocation is found, it transitions to blockStatePending.
	blockStateHead

	// blockStateHeadUnmanaged is the state of a block which starts an unmanaged (`malloc`ed by C) allocation.
	// It is never marked, and is ignored by the sweep pass.
	blockStateHeadUnmanaged

	// blockStateHeadNoPointers is the state of a block which starts an allocation containing no pointers.
	// When a reference to the allocation is found, it transitions directly to blockStateMarkedNoPointers.
	blockStateHeadNoPointers

	// blockStateTail is the state of a block continuing an allocation.
	// It must follow either a head (blockStateHead/blockStateHeadUnmanaged/blockStateHeadNoPointers) or another tail.
	blockStateTail

	// blockStatePending is the state of a head which has been marked but not yet scanned.
	// After it is scanned, it transitions to blockStateMarked.
	blockStatePending

	// blockStateMarked is the state of a head which has been marked and scanned.
	// After the sweep phase, it transitions back to blockStateHead.
	blockStateMarked

	// blockStateMarkedNoPointers is the state of the head of a pointerless allocation which has been marked.
	// After the sweep phase, it transitions back to blockStateHeadNoPointers.
	blockStateMarkedNoPointers
)

// zeroSizedAlloc is just a sentinel that gets returned when allocating 0 bytes.
var zeroSizedAlloc uint8

// scanTyped reads all pointers from start to end (exclusive) and if they look
// like a heap pointer and are unmarked, marks them. The start and end parameters
// must be valid pointers and must be aligned. It uses the provided type to skip
// non-pointery memory.
func scanTyped(start, end uintptr, typ unsafe.Pointer) {
	if typ == nil {
		return
	}

	// Decode the size and bitmap of the type.
	size := *(*uintptr)(typ)
	bitmap := uintptr(typ) + ptrSize

	// Reduce the end bound to avoid reading too far on platforms where pointer alignment is smaller than pointer size.
	// If the size of the range is 0, then end will be slightly below start after this.
	end -= unsafe.Sizeof(end) - unsafe.Alignof(end)

	var i uintptr
	var w uint8
	for addr := start; addr < end; addr += unsafe.Alignof(addr) {
		if i > size {
			// Loop back to the start of the bitmap.
			i = 0
		}

		if i%8 == 0 {
			// Load the next word in the bitmap.
			w = *(*uint8)(bitmap + (i / 8))
		}

		// Check if this address may contain a pointer.
		isPtr := w&1 != 0
		w >>= 1
		i++
		if !isPtr {
			continue
		}

		// Mark the pointer stored at this address.
		root := *(*uintptr)(unsafe.Pointer(addr))
		markRoot(addr, root)
	}
}

// markRoots reads all pointers from start to end (exclusive) and if they look
// like a heap pointer and are unmarked, marks them. The start and end parameters
// must be valid pointers and must be aligned.
func markRoots(start, end uintptr) {
	if gcDebug {
		println("mark from", start, "to", end, int(end-start))
	}
	if gcAsserts {
		if start >= end {
			runtimePanic("gc: unexpected range to mark")
		}
		if start%unsafe.Alignof(start) != 0 {
			runtimePanic("gc: unaligned start pointer")
		}
		if end%unsafe.Alignof(end) != 0 {
			runtimePanic("gc: unaligned end pointer")
		}
	}

	// Reduce the end bound to avoid reading too far on platforms where pointer alignment is smaller than pointer size.
	// If the size of the range is 0, then end will be slightly below start after this.
	end -= unsafe.Sizeof(end) - unsafe.Alignof(end)

	for addr := start; addr < end; addr += unsafe.Alignof(addr) {
		root := *(*uintptr)(unsafe.Pointer(addr))
		markRoot(addr, root)
	}
}

// mark a GC root at the address addr.
func markRoot(addr, root uintptr) {
	if !looksLikePointer(root) {
		// This is definitely not a pointer.
		return
	}

	// Find the block.
	block := blockFromAddr(root)
	if block.meta().state() == blockStateFree {
		// The to-be-marked object doesn't actually exist.
		// This could either be a dangling pointer (oops!) but most likely
		// just a false positive.
		return
	}
	head := block.findHead()
	meta := head.meta()
	switch meta.state() {
	case blockStateHead:
		if gcDebug {
			println("found unmarked pointer", root, "at address", addr)
		}

		// Change the state to pending.
		meta.setState(blockStatePending)

		// Update the pending scan tree.
		b := head
		for {
			meta := b.meta()
			if meta.maybeHasChildPending() {
				// This is already up to date.
				break
			}

			// Flag this as having pending children that need to be scanned.
			meta.setChildPending()

			// Move up to the parent block.
			if b == 0 {
				break
			}
			b = (b - 1) / 2
		}

	case blockStateHeadNoPointers:
		if gcDebug {
			println("found unmarked pointer", root, "to pointerless allocation at address", addr)
		}

		// Mark the allocation.
		// No scanning is required.
		meta.setState(blockStateMarkedNoPointers)

	case blockStateHeadUnmanaged:
		if gcDebug {
			println("found pointer", root, "to unmanaged allocation at address", addr)
		}

		// This memory is handled manually by malloc and free.
		// Ignore it.
	}
}

// looksLikePointer returns whether this could be a pointer. Currently, it
// simply returns whether it lies anywhere in the heap. Go allows interior
// pointers so we can't check alignment or anything like that.
func looksLikePointer(ptr uintptr) bool {
	return ptr >= heapStart && ptr < uintptr(metadataStart)
}

// findHead returns the head (first block) of an object, assuming the block
// points to an allocated object. It returns the same block if this block
// already points to the head.
func (b gcBlock) findHead() gcBlock {
	for b.meta().state() == blockStateTail {
		b--
	}
	if gcAsserts {
		switch b.meta().state() {
		case blockStateHead, blockStateHeadNoPointers, blockStateHeadUnmanaged:
		default:
			runtimePanic("gc: found tail without head")
		}
	}
	return b
}

// findMem reserves a contiguous range of n blocks.
// The allocation is initialized in an unmanaged state.
// It may run a garbage collection cycle if needed.
func findMem(n uintptr) (gcBlock, bool) {
	// Try to allocate immediately.
	if block, ok := tryFindMem(n); ok {
		return block
	}

	// Run the garbage collector.
	GC()

	for {
		// Try to allocate with the current heap size.
		if block, ok := tryFindMem(n); ok {
			return block
		}

		// Try to grow the heap.
		if !growHeap() {
			// Unfortunately the heap could not be increased. This
			// happens on baremetal systems for example (where all
			// available RAM has already been dedicated to the heap).
			runtimePanic("out of memory")
		}
	}
}

// tryFindMem searches for a contiguous range of n blocks.
// The allocation is initialized in an unmanaged state.
func tryFindMem(n uintptr) (gcBlock, bool) {
	// Traverse backwards in case something was freed since the last call.
	for nextAlloc > 0 && (nextAlloc-1).meta.state() == blockStateFree {
		nextAlloc--
	}

	// Loop through the heap searching for memory.
	index := nextAlloc
	var freeBlocks uintptr
	var start gcBlock
	for {
		if index == endBlock {
			// Loop back around to the start.
			index = 0
			freeBlocks = 0
		}

		// Check if this block is free.
		if *index.meta() == 0 {
			// This block is free.
			if freeBlocks == 0 {
				// This is the start of a new range.
				start = index
			}
			freeBlocks++
			if freeBlocks == n {
				// A sufficiently large range has been found.
				break
			}
		} else {
			// This block terminates the free range.
			freeBlocks = 0
		}

		// Move to the next block.
		index++
		if index == nextAlloc {
			// The entire heap was scanned, but nothing was found.
			return 0, false
		}
	}

	// Set up the GC metadata.
	start.meta().setState(blockStateHeadUnmanaged)
	for i := start + 1; i <= index; i++ {
		i.meta().setState(blockStateTail)
	}

	// Update nextAlloc for next time.
	nextAlloc = index + 1

	return start, true
}

func allocUnmanaged(size uintptr) unsafe.Pointer {
	if size == 0 {
		return nil
	}

	block, ok := findMem(sizeBlocks(size))
	if !ok {
		return nil
	}

	return unsafe.Pointer(block.data())
}

func freeUnmanaged(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	findUnmanagedAlloc(ptr).free()
}

func reallocUnmanaged(ptr unsafe.Pointer, size uintptr) unsafe.Pointer {
	if ptr == nil {
		// This is just equivallent to malloc.
		return allocUnmanaged(size)
	}

	// Calculate the target size in blocks.
	sizeBlocks := sizeBlocks(size)

	// Attempt to adjust the size of the allocation.
	block := findUnmanagedAlloc(ptr)
	end := block.end()
	oldSizeBlocks := uintptr(end-block) + 1
	if oldSizeBlocks >= sizeBlocks {
		// Shrink the allocation.
		memzero(unsafe.Pointer((b + gcBlock(sizeBlocks)).meta()), oldSizeBlocks-sizeBlocks)
		return ptr
	}

	// Check for additional space.
	for i := block + sizeBlocks - 1; i > end; i++ {
		if *i.meta() != 0 {
			// This is not available.
			// Allocate new memory and copy.
			newBlock, ok := findMem(sizeBlocks)
			if !ok {
				return nil
			}
			newAlloc := unsafe.Pointer(newBlock.data())
			memcpy(newAlloc, ptr, oldSizeBlocks*bytesPerBlock)
			block.free()
			return newAlloc
		}
	}

	// Extend the allocation.
	for i := end + 1; i < block+sizeBlocks; i++ {
		i.meta().setState(blockStateTail)
	}

	return ptr
}

func findUnmanagedAlloc(ptr unsafe.Pointer) gcBlock {
	if !looksLikePointer(uinptr(ptr)) {
		runtimePanic("invalid pointer")
	}
	block := blockFromAddr(uintptr(ptr))
	if ptr != unsafe.Pointer(block.data()) {
		runtimePanic("pointer is not the start of a block")
	}
	switch block.meta().state() {
	case blockStateHeadUnmanaged:
		return block

	case blockStateFree:
		runtimePanic("pointer to freed memory")
	case blockStateTail:
		runtimePanic("pointer points into tail of allocation")
	default:
		runtimePanic("pointer references managed memory")
	}

	panic("unreachable")
}

// GC performs a garbage collection cycle.
func GC() {
	if gcDebug {
		println("running collection cycle...")
	}

	// Mark phase: mark all reachable objects, recursively.
	markStack()
	markGlobals()

	if baremetal && hasScheduler {
		// Channel operations in interrupts may move task pointers around while we are marking.
		// Therefore we need to scan the runqueue seperately.
		var markedTaskQueue task.Queue
	runqueueScan:
		for !runqueue.Empty() {
			// Pop the next task off of the runqueue.
			t := runqueue.Pop()

			// Mark the task if it has not already been marked.
			markRoot(uintptr(unsafe.Pointer(&runqueue)), uintptr(unsafe.Pointer(t)))

			// Push the task onto our temporary queue.
			markedTaskQueue.Push(t)
		}

		finishMark()

		// Restore the runqueue.
		i := interrupt.Disable()
		if !runqueue.Empty() {
			// Something new came in while finishing the mark.
			interrupt.Restore(i)
			goto runqueueScan
		}
		runqueue = markedTaskQueue
		interrupt.Restore(i)
	} else {
		finishMark()
	}

	// Sweep phase: free all non-marked objects and unmark marked objects for
	// the next collection cycle.
	sweep()

	// Show how much has been sweeped, for debugging.
	if gcDebug {
		dumpHeap()
	}
}

func finishMark() {
	endBlock := endBlock
	var i gcBlock
	for {
		// Traverse down the tree.
		if l := 2*i + 1; l < endBlock && l.meta().maybeHasChildPending() {
			// Move down to the left child.
			i = l
			continue
		}
		if r := 2*i + 2; r < endBlock && r.meta().maybeHasChildPending() {
			// Move down to the right child.
			i = r
			continue
		}

		// Check the current node.
		meta := i.meta()
		if meta.state() == blockStatePending {
			// Scan this allocation.
			i.scan()
			meta.setState(blockStateMarked)
			continue
		}

		// There is nothing to see here.
		meta.clearChildPending()

		// Move up the tree.
		if i == 0 {
			// There is nothing left to do.
			return
		}
		i = (i - 1) / 2
	}
}

func sweep() {
	for i := gcBlock(0); i < endBlock; i++ {
		meta := i.meta()
		switch meta.state() {
		case blockStateHead, blockStateHeadNoPointers:
			// This allocation has not been marked.
			i.free()

		case blockStateHeadUnmanaged:
			// Ignore this.

		case blockStateMarked:
			// Return this to a normal head state.
			meta.setState(blockStateHead)

		case blockStateMarkedNoPointers:
			// Return this to a normal head state.
			meta.setState(blockStateHeadNoPointers)

		case blockStateTail, blockStateFree:
			// Nothing to do.

		case blockStatePending:
			if gcAsserts {
				runtimePanic("found a pending allocation that was not scanned")
			}
		}
	}
	nextAlloc = 0
}

func (b gcBlock) free() {
	memzero(unsafe.Pointer(b.meta()), uintptr(b.end()-b)+1)
}

func sizeBlocks(size uintptr) uintptr {
	return (size + (bytesPerBlock - 1)) / bytesPerBlock
}
