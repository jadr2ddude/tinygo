package goroutine

import (
	"fmt"
	"os"
	"tinygo.org/x/go-llvm"
)

func loadTestIR(path string) (llvm.Module, error) {
	ctx := llvm.NewContext()
	buf, err := llvm.NewMemoryBufferFromFile(path)
	os.Stat(path) // make sure this file is tracked by `go test` caching
	if err != nil {
		return llvm.Module{}, fmt.Errorf("could not read file %s: %v", path, err)
	}
	mod, err := ctx.ParseIR(buf)
	if err != nil {
		return llvm.Module{}, fmt.Errorf("could not load module %s:\n%v", path, err)
	}
	return mod, nil
}

// functionNames creates a slice of the names of the functions in the input slice.
// This is used to make test output more readable.
func functionNames(funcs []llvm.Value) []string {
	out := make([]string, len(funcs))
	for i, f := range funcs {
		out[i] = f.Name()
	}
	return out
}
