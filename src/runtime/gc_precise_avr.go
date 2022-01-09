//go:build gc.precise && avr
// +build gc.precise,avr

package runtime

// typePtr returns the type pointer for an allocation.
// The provided block is assumed to be blockStatePending.
func (b gcBlock) typePtr(end gcBlock) unsafe.Pointer {
	if end-b >= 2*ptrSize {
		// If the allocation is 4 or more blocks in size,
		// the type pointer is stored in the "extra" bits
		// of the first 4 blocks.
		return b.typePtrBig
	}

	// Load the pointer from the end of the last block.
	return *(*unsafe.Pointer)(unsafe.Pointer(&end.data()[bytesPerBlock-ptrSize]))
}

// typePtrBig returns the type pointer for a large allocation.
// The type pointer is stored using the extra bits from each of the first 4 blocks.
func (b gcBlock) typePtrBig() unsafe.Pointer {
	var w uintptr
	for i := gcBlock(0); i < 2*ptrSize; i++ {
		w |= (b + i).meta().extra() << (4 * i)
	}
	return unsafe.Pointer(w)
}

// scan the pointers within the allocation and mark them.
func (b gcBlock) scan() {
	// Find the end of the allocation.
	end := b.end()

	// Obtain the type pointer.
	typePtr := b.typePtr(end)

	// Get the start and end pointers.
	start := uintptr(unsafe.Pointer(b.data()))
	endData := end.data()
	end := uintptr(unsafe.Pointer(endData)) + uintptr(len(endData))

	// Scan the memory range using the type.
	scanTyped(start, end, typePtr)
}

func sizeBlocksTyped(size uintptr) uintptr {
	if size < ((2*ptrSize)-1)*bytesPerBlock {
		size += ptrSize
	}

	return sizeBlocks(size)
}
