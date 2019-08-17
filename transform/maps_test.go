package transform

import (
	"testing"
)

func TestOptimizeMaps(t *testing.T) {
	t.Parallel()
	testTransform(t, "testdata/maps", OptimizeMaps)
}
