package goroutine

import (
	"reflect"
	"testing"
	"tinygo.org/x/go-llvm"
)

func TestFindBlockingFunctions(t *testing.T) {
	t.Parallel()

	// Load test input IR.
	mod, err := loadTestIR("testdata/callgraph.ll")
	if err != nil {
		t.Errorf("failed to load test input IR: %v", err)
		return
	}

	// Find all of the functions used in the test.
	var expectFuncs []llvm.Value
	for _, name := range []string{"pause", "z", "y", "x"} {
		fn := mod.NamedFunction(name)
		if fn.IsNil() {
			t.Errorf("missing function in test input: %s", name)
			return
		}
		expectFuncs = append(expectFuncs, fn)
	}

	// Build the expected call graph.
	expectgraph := digraph{
		0: nil,
		1: {0, 2},
		2: {1},
		3: {2},
	}

	// Find all of the blocking functions.
	funcs, callgraph := findBlockingFunctions(mod.NamedFunction("pause"))

	// Check the function list.
	if !reflect.DeepEqual(funcs, expectFuncs) {
		t.Errorf("expected function list %v; got %v", functionNames(expectFuncs), functionNames(funcs))
	}

	// Check the call graph.
	if !graphEqual(callgraph, expectgraph) {
		t.Errorf("expected call graph %v; got %v", expectgraph, callgraph)
	}
}
