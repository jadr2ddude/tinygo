package goroutine

import (
	"sort"

	"github.com/RoaringBitmap/roaring"
	"tinygo.org/x/go-llvm"
)

// subtypes returns a list of the types directly contained in a composite type.
// There may be duplicates.
// If the type is primitive, this returns nil.
// If the type is an array or vector, this returns a slice containing one copy of the element type.
// If the type is a pointer and expandPtr is true, this returns the pointed-to value.
// If the type is a pointer and expandPtr is false, this treats the pointer as a primitive type.
// If the type is a struct, returns the field types.
// If the type is a function, returns the return type (omitted if void) followed by the parameter types.
func subtypes(t llvm.Type, expandPtr bool) []llvm.Type {
	switch t.TypeKind() {
	case llvm.FloatTypeKind, llvm.DoubleTypeKind, llvm.X86_FP80TypeKind, llvm.FP128TypeKind, llvm.PPC_FP128TypeKind, llvm.IntegerTypeKind:
		// The type is primitive.
		return nil
	case llvm.StructTypeKind:
		// Decompose the struct type into its elements.
		return t.StructElementTypes()
	case llvm.PointerTypeKind:
		if !expandPtr {
			return nil
		}
		fallthrough
	case llvm.ArrayTypeKind, llvm.VectorTypeKind:
		// Extract the element type.
		return []llvm.Type{t.ElementType()}
	case llvm.FunctionTypeKind:
		// Extract parameter types.
		subtypes := t.ParamTypes()

		if returnType := t.ReturnType(); returnType.TypeKind() != llvm.VoidTypeKind {
			// Prepend the return type.
			subtypes = append([]llvm.Type{returnType}, subtypes...)
		}

		return subtypes
	default:
		panic("unrecognized type kind")
	}
}

// expandTypeList expands a list of types to include all subtypes.
// If expandPtr is false, pointed-to types are not added.
// The input list must not have duplicates.
func expandTypeList(types []llvm.Type, expandPtr bool) []llvm.Type {
	typeList := append([]llvm.Type(nil), types...)
	typesFound := map[llvm.Type]struct{}{}
	for _, t := range types {
		typesFound[t] = struct{}{}
	}
	for i := 0; i < len(typeList); i++ {
		for _, st := range subtypes(typeList[i], expandPtr) {
			if _, ok := typesFound[st]; ok {
				// This type was already encountered.
				continue
			}

			typeList = append(typeList, st)
			typesFound[st] = struct{}{}
		}
	}
	return typeList
}

// typeKindPriority is a helper that assigns a priority number to a type kind so that it can be sorted.
func typeKindPriority(k llvm.TypeKind) int {
	for i, v := range []llvm.TypeKind{llvm.StructTypeKind, llvm.VectorTypeKind, llvm.ArrayTypeKind, llvm.FP128TypeKind, llvm.DoubleTypeKind, llvm.FloatTypeKind, llvm.PPC_FP128TypeKind, llvm.X86_FP80TypeKind, llvm.IntegerTypeKind, llvm.PointerTypeKind, llvm.FunctionTypeKind} {
		if k == v {
			return i
		}
	}
	panic("unrecognized type")
}

// maxTypeDepth returns the depth of the deepest recursive subtype.
// Pointers, primitive types, or empty structs will return 1.
// A struct will return 1 beyond the depth of its deepest element.
// A vector or array will return 1 beyond the depth of its element type.
func maxTypeDepth(t llvm.Type) int {
	var max int
	for _, st := range subtypes(t, false) {
		depth := maxTypeDepth(st)
		if depth > max {
			max = depth
		}
	}
	return max + 1
}

// compareTypes compares types for sort ordering.
// Zero indicates that the types are equal or identical.
// A negative value indicates that x comes before y.
// A positive value indicates that y comes before x.
func compareTypes(x, y llvm.Type) int {
	if x == y {
		// The types are equal.
		return 0
	}

	if xdepth, ydepth := maxTypeDepth(x), maxTypeDepth(y); xdepth != ydepth {
		// Compare the type depths.
		return ydepth - xdepth
	}

	if xkind, ykind := x.TypeKind(), y.TypeKind(); xkind != ykind {
		// Compare the type kinds.
		return typeKindPriority(xkind) - typeKindPriority(ykind)
	}

	switch x.TypeKind() {
	case llvm.StructTypeKind:
		if xpacked, ypacked := x.IsStructPacked(), y.IsStructPacked(); xpacked != ypacked {
			// Compare struct packing.
			// Packed structs come after unpacked structs.
			if xpacked {
				return 1
			} else {
				return -1
			}
		}
	case llvm.ArrayTypeKind:
		if xlen, ylen := x.ArrayLength(), y.ArrayLength(); xlen != ylen {
			// Compare array lengths.
			return ylen - xlen
		}
	case llvm.VectorTypeKind:
		if xsize, ysize := x.VectorSize(), y.VectorSize(); xsize != ysize {
			// Compare vector sizes.
			return ysize - xsize
		}
	case llvm.IntegerTypeKind:
		if xbits, ybits := x.IntTypeWidth(), y.IntTypeWidth(); xbits != ybits {
			// Compare integer type widths.
			return ybits - xbits
		}
	case llvm.PointerTypeKind:
		if xspace, yspace := x.PointerAddressSpace(), y.PointerAddressSpace(); xspace != yspace {
			// Compare pointer address spaces.
			return yspace - xspace
		}
	case llvm.FunctionTypeKind:
		if xret, yret := x.ReturnType().TypeKind() != llvm.VoidTypeKind, y.ReturnType().TypeKind() != llvm.VoidTypeKind; xret != yret {
			// Compare presence of a return value.
			// Function types with a return value come first.
			if xret {
				return -1
			} else {
				return 1
			}
		}
	}

	// Decompose type and compare subtypes.
	xtypes, ytypes := subtypes(x, true), subtypes(y, true)
	if len(xtypes) != len(ytypes) {
		// Compare the number of subtypes or struct fields.
		return len(ytypes) - len(xtypes)
	}
	for i := range xtypes {
		cmp := compareTypes(xtypes[i], ytypes[i])
		if cmp != 0 {
			return cmp
		}
	}

	// These types are renamed versions of each other.
	return 0
}

// sortTypes sorts types such that they are in a mostly consistent order.
// Composite types will always come before their element types.
// Equivalent types will remain in their original order.
func sortTypes(types []llvm.Type) {
	sort.SliceStable(types, func(i, j int) bool {
		return compareTypes(types[i], types[j]) > 0
	})
}

// buildTypeDAG builds a directed acyclic graph from a type list, with vertices from composite types to their component types.
// The vertex indices in the resulting graph correspond to the indices of the input type list.
// The input type list must be expanded, such as with expandTypeList.
// The resulting graph is normalized for convenience.
func buildTypeDAG(types []llvm.Type) digraph {
	// Build an index of the type list.
	typeIndex := map[llvm.Type]int{}
	for i, t := range types {
		typeIndex[t] = i
	}

	// Build the type graph.
	graph := make(digraph, len(types))
	for i, t := range types {
		found := map[llvm.Type]struct{}{}
		for _, st := range subtypes(t, false) {
			if _, ok := found[st]; ok {
				// This subtype was already found.
				continue
			}

			// Add an edge to the subtype.
			graph[i] = append(graph[i], typeIndex[st])
			found[st] = struct{}{}
		}
	}

	// Normalize the type graph.
	graph.normalize()

	return graph
}

// layoutAllocas maps allocas into slots.
// Each index in the slots list contains the used lifetime of the slot.
// If the slots are not sufficient to map all allocas, a new slot will be allocated.
func layoutAllocas(slots []roaring.Bitmap, allocas []roaring.Bitmap) (map[int]int, []roaring.Bitmap) {
	type slotEntry struct {
		index    int
		allocas  []int
		lifetime roaring.Bitmap
	}

	// Build slot entries.
	slotEntries := make([]slotEntry, len(slots))
	for i, l := range slots {
		slotEntries[i] = slotEntry{
			index:    i,
			lifetime: *l.Clone(),
		}
	}

	// Build allocation mapping queue, sorted in descending cardinality order.
	allocaQueue := make([]int, len(allocas))
	for i := range allocas {
		allocaQueue[i] = i
	}
	sort.SliceStable(allocaQueue, func(i, j int) bool {
		return allocas[allocaQueue[i]].GetCardinality() > allocas[allocaQueue[j]].GetCardinality()
	})

	// Sort slots in order of descending cardinality.
	sort.SliceStable(slotEntries, func(i, j int) bool {
		return slotEntries[i].lifetime.GetCardinality() > slotEntries[j].lifetime.GetCardinality()
	})

	for _, i := range allocaQueue {
		// Get the lifetime of this alloca.
		lifetime := allocas[i]

		// Search for a valid slot.
		slot := 0
		for ; slot < len(slotEntries); slot++ {
			if !slotEntries[slot].lifetime.Intersects(&lifetime) {
				// The lifetimes do not overlap.
				// Therefore, the slot can be merged.
				break
			}
		}

		if slot == len(slotEntries) {
			// This alloca intersects all available slots.
			// Create a new slot for it.
			slotEntries = append(slotEntries, slotEntry{
				index:    len(slotEntries),
				allocas:  []int{i},
				lifetime: *lifetime.Clone(),
			})
			continue
		}

		// Merge the alloca into the selected slot.
		slotEntries[slot].lifetime.Or(&lifetime)
		slotEntries[slot].allocas = append(slotEntries[slot].allocas, i)

		for slot > 1 && slotEntries[slot-1].lifetime.GetCardinality() < slotEntries[slot].lifetime.GetCardinality() {
			// The slot has gained cardinality, and now needs to be moved earlier to restore the sorting.
			slotEntries[slot-1], slotEntries[slot] = slotEntries[slot], slotEntries[slot-1]
			slot--
		}
	}

	// Build map of allocas to slots.
	allocaSlots := map[int]int{}
	for _, s := range slotEntries {
		for _, a := range s.allocas {
			allocaSlots[a] = s.index
		}
	}

	// Extract resulting slot lifetimes.
	slotLifetimes := make([]roaring.Bitmap, len(slotEntries))
	for _, ent := range slotEntries {
		slotLifetimes[ent.index] = ent.lifetime
	}

	return allocaSlots, slotLifetimes
}
