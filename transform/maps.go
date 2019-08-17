package transform

import (
	"github.com/tinygo-org/tinygo/llvmutil"
	"tinygo.org/x/go-llvm"
)

// OptimizeMaps eliminates created but unused maps.
//
// In the future, this should statically allocate created but never modified
// maps. This has not yet been implemented, however.
func OptimizeMaps(mod llvm.Module) {
	hashmapMake := mod.NamedFunction("runtime.hashmapMake")
	if hashmapMake.IsNil() {
		// nothing to optimize
		return
	}

	hashmapBinarySet := mod.NamedFunction("runtime.hashmapBinarySet")
	hashmapStringSet := mod.NamedFunction("runtime.hashmapStringSet")

	for _, makeInst := range llvmutil.GetUses(hashmapMake) {
		updateInsts := []llvm.Value{}
		unknownUses := false // are there any uses other than setting a value?

		for _, use := range llvmutil.GetUses(makeInst) {
			if use := use.IsACallInst(); !use.IsNil() {
				switch use.CalledValue() {
				case hashmapBinarySet, hashmapStringSet:
					updateInsts = append(updateInsts, use)
				default:
					unknownUses = true
				}
			} else {
				unknownUses = true
			}
		}

		if !unknownUses {
			// This map can be entirely removed, as it is only created but never
			// used.
			for _, inst := range updateInsts {
				inst.EraseFromParentAsInstruction()
			}
			makeInst.EraseFromParentAsInstruction()
		}
	}
}
