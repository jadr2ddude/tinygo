package transform_test

import (
	"testing"

	"github.com/tinygo-org/tinygo/transform"
)

func TestSoftStack(t *testing.T) {
	t.Parallel()
	testTransform(t, "testdata/softstack", transform.SoftStackify)
}
