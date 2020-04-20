package goroutine

import "tinygo.org/x/go-llvm"

// findBlockingFunctions finds all blocking functions in the module.
// A function is considered blocking if it calls the pause intrinsic or a blocking function.
// Returns a list of function values and a call graph.
// The call graph vertices match the indices of the function list.
// An edge in the call graph represents the existence of a call to the head by the tail.
// The callee graph is normalized.
func findBlockingFunctions(pause llvm.Value) ([]llvm.Value, digraph) {
	functions := []llvm.Value{pause}
	funcIndex := map[llvm.Value]int{pause: 0}
	var callerGraph digraph
	for i := 0; i < len(functions); i++ {
		fn := functions[i]
		var callers []int
		callersFound := map[llvm.Value]struct{}{}
		for use := fn.FirstUse(); !use.IsNil(); use = use.NextUse() {
			user := use.User()
			switch {
			case user.IsACallInst().IsNil():
				// The use is not a call instruction.
			case user.CalledValue() != fn:
				// The function is passsed as an argument in the call.
			default:
				// The function is called.

				// Find the function containing the call instruction.
				caller := user.InstructionParent().Parent()

				if _, ok := callersFound[caller]; ok {
					// This caller is already in the list.
					continue
				}

				if _, ok := funcIndex[caller]; !ok {
					// This function has not been previously encountered.
					// Add it to the index and function list.
					funcIndex[caller] = len(functions)
					functions = append(functions, caller)
				}

				// Add the caller to the list.
				callers = append(callers, funcIndex[caller])
				callersFound[caller] = struct{}{}
			}
		}
		callerGraph = append(callerGraph, callers)
	}

	// Turn the caller graph into a callee graph.
	calleeGraph := callerGraph.reverse()

	return functions, calleeGraph
}
