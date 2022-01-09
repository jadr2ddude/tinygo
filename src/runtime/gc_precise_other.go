//go:build gc.precise && !avr
// +build gc.precise,!avr

package runtime

// scan the pointers within the allocation and mark them.
func (b gcBlock) scan() {
	for {
		// A bitmap is of pointers is stored within the extra metadata bits.
		mask := b.meta().extra()
		data := (*[wordsPerBlock]uintptr)(unsafe.Pointer(b.data()))
		for i := range data {
			if mask&(1<<i) == 0 {
				// This is not a pointer.
				// Skip it.
				continue
			}

			markRoot(uintptr(unsafe.Pointer(&data[i])), data[i])
		}

		// Move on to the next block.
		b++
		if b >= endBlock || b.meta().state() != blockStateTail {
			break
		}
	}
}

func sizeBlocksTyped(size uintptr) uintptr {
	// There is no added overhead.
	return sizeBlocks(size)
}
