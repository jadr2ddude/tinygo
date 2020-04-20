package goroutine

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/RoaringBitmap/roaring"
	"tinygo.org/x/go-llvm"
)

func TestSortTypes(t *testing.T) {
	t.Parallel()

	// Create an LLVM context to build the types with.
	ctx := llvm.NewContext()
	defer ctx.Dispose()

	// Create some basic types.
	f32 := ctx.FloatType()
	f64 := ctx.DoubleType()
	i1 := ctx.Int1Type()
	i16 := ctx.Int16Type()
	i8 := ctx.Int8Type()
	i8ptr := llvm.PointerType(i8, 0)
	i8progmemptr := llvm.PointerType(i8, 1)
	f32ptr := llvm.PointerType(f32, 0)

	// Create some basic composites.
	cplx64 := llvm.StructType([]llvm.Type{f32, f32}, false)
	cplx128 := llvm.StructType([]llvm.Type{f64, f64}, false)
	f32triplet := llvm.ArrayType(f32, 3)
	i16x8vec := llvm.VectorType(i16, 8) // One ARM NEON quadword register.

	// Create a pointer to a composite.
	i16x8vecptr := llvm.PointerType(i16x8vec, 0)

	// Create a deeper composite.
	f32mat3x3 := llvm.ArrayType(f32triplet, 3)

	// Create a pointer to a composite.
	f32mat3x3ptr := llvm.PointerType(f32mat3x3, 0)

	// Put all of the types into a sorted list.
	types := []llvm.Type{f32mat3x3, i16x8vec, f32triplet, cplx128, cplx64, f64, f32, i16, i8, i1, i8progmemptr, f32mat3x3ptr, i16x8vecptr, f32ptr, i8ptr}

	rng := rand.New(rand.NewSource(14))

	for i := 0; i < 64; i++ {
		// Copy the type list and shuffle it.
		shuffledTypes := append([]llvm.Type(nil), types...)
		rng.Shuffle(len(shuffledTypes), func(i, j int) {
			shuffledTypes[j], shuffledTypes[i] = shuffledTypes[i], shuffledTypes[j]
		})

		// Sort the shuffled type list.
		sortTypes(shuffledTypes)

		// Verify that the shuffled list has been returned to its original order.
		if !reflect.DeepEqual(types, shuffledTypes) {
			t.Errorf("expected %v; got %v", types, shuffledTypes)
		}
	}
}

func buildBitmap(nums ...uint32) roaring.Bitmap {
	var bitmap roaring.Bitmap
	bitmap.AddMany(nums)
	return bitmap
}

func TestLayoutAllocas(t *testing.T) {
	cases := []struct {
		name     string
		slots    []roaring.Bitmap
		allocas  []roaring.Bitmap
		outmap   map[int]int
		outslots [][]uint32
	}{
		{
			name:     "empty",
			outmap:   map[int]int{},
			outslots: [][]uint32{},
		},
		{
			name: "simple",
			slots: []roaring.Bitmap{
				buildBitmap(),
			},
			allocas: []roaring.Bitmap{
				buildBitmap(1),
				buildBitmap(2),
				buildBitmap(3),
			},
			outmap: map[int]int{
				0: 0,
				1: 0,
				2: 0,
			},
			outslots: [][]uint32{
				[]uint32{1, 2, 3},
			},
		}, {
			name: "noslots",
			allocas: []roaring.Bitmap{
				buildBitmap(1),
				buildBitmap(2),
				buildBitmap(3),
			},
			outmap: map[int]int{
				0: 0,
				1: 0,
				2: 0,
			},
			outslots: [][]uint32{
				[]uint32{1, 2, 3},
			},
		},
	}

	for _, c := range cases {
		caseDat := c
		t.Run(caseDat.name, func(t *testing.T) {
			t.Parallel()
			outmap, outslotsraw := layoutAllocas(caseDat.slots, caseDat.allocas)
			outslots := make([][]uint32, len(outslotsraw))
			for i, s := range outslotsraw {
				outslots[i] = s.ToArray()
			}
			if !reflect.DeepEqual(outmap, caseDat.outmap) {
				t.Errorf("expected alloca-slot map %v; got %v", caseDat.outmap, outmap)
			}
			if !reflect.DeepEqual(outslots, caseDat.outslots) {
				t.Errorf("expected slot allocation %v; got %v", caseDat.outslots, outslots)
			}
		})
	}
}
