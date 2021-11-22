//go:build baremetal && !tinygo.wasm
// +build baremetal,!tinygo.wasm

// Automatically generated file. DO NOT EDIT.
// This file implements standins for non-native atomics using critical sections.

package runtime

import (
	"runtime/interrupt"
	"unsafe"
)

// Documentation:
// * https://llvm.org/docs/Atomics.html
// * https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
//
// Some atomic operations are emitted inline while others are emitted as libcalls.
// How many are emitted as libcalls depends on the MCU arch and core variant.

// Weirdly sized atomics (size is a compile-time constant).

//export __atomic_load
func __atomic_load(size uintptr, ptr, ret unsafe.Pointer, ordering uintptr) {
	mask := interrupt.Disable()
	memcpy(ret, ptr, size)
	interrupt.Restore(mask)
}

//export __atomic_store
func __atomic_store(size uintptr, ptr, val unsafe.Pointer, ordering uintptr) {
	mask := interrupt.Disable()
	memcpy(ptr, val, size)
	interrupt.Restore(mask)
}

//export __atomic_exchange
func __atomic_exchange(size uintptr, ptr, val, ret unsafe.Pointer, ordering uintptr) {
	mask := interrupt.Disable()
	memcpy(ptr, val, size)
	memcpy(ret, ptr, size)
	interrupt.Restore(mask)
}

//export __atomic_compare_exchange
func __atomic_compare_exchange(size uintptr, ptr, expected, desired unsafe.Pointer, successOrder, failureOrder uintptr) bool {
	mask := interrupt.Disable()
	ok := memequal(ptr, expected, size)
	if ok {
		memcpy(ptr, desired, size)
	}
	interrupt.Restore(mask)
	return ok
}

// 8-bit atomics.

//export __atomic_load_1
func __atomic_load_1(ptr *uint8, ordering uintptr) uint8 {
	// The LLVM docs for this have a typo saying that there is a val argument after the pointer.
	// That is a typo, and the GCC docs omit it.
	mask := interrupt.Disable()
	val := *ptr
	interrupt.Restore(mask)
	return val
}

//export __atomic_store_1
func __atomic_store_1(ptr *uint8, val uint8, ordering uintptr) {
	mask := interrupt.Disable()
	*ptr = val
	interrupt.Restore(mask)
}

//go:inline
func doAtomicCAS8(ptr *uint8, expected, desired uint8) uint8 {
	mask := interrupt.Disable()
	old := *ptr
	if old == expected {
		*ptr = desired
	}
	interrupt.Restore(mask)
	return old
}

//export __sync_val_compare_and_swap_1
func __sync_val_compare_and_swap_1(ptr *uint8, expected, desired uint8) uint8 {
	return doAtomicCAS8(ptr, expected, desired)
}

//export __atomic_compare_exchange_1
func __atomic_compare_exchange_1(ptr, expected *uint8, desired uint8, successOrder, failureOrder uintptr) bool {
	exp := *expected
	old := doAtomicCAS8(ptr, exp, desired)
	return old == exp
}

//go:inline
func doAtomicSwap8(ptr *uint8, new uint8) uint8 {
	mask := interrupt.Disable()
	old := *ptr
	*ptr = new
	interrupt.Restore(mask)
	return old
}

//export __sync_lock_test_and_set_1
func __sync_lock_test_and_set_1(ptr *uint8, new uint8) uint8 {
	return doAtomicSwap8(ptr, new)
}

//export __atomic_exchange_1
func __atomic_exchange_1(ptr *uint8, new uint8, ordering uintptr) uint8 {
	return doAtomicSwap8(ptr, new)
}

//go:inline
func doAtomicAdd8(ptr *uint8, value uint8) (old, new uint8) {
	mask := interrupt.Disable()
	old = *ptr
	new = old + value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_add_1
func __atomic_fetch_add_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	old, _ := doAtomicAdd8(ptr, value)
	return old
}

//export __sync_fetch_and_add_1
func __sync_fetch_and_add_1(ptr *uint8, value uint8) uint8 {
	old, _ := doAtomicAdd8(ptr, value)
	return old
}

//export __atomic_add_fetch_1
func __atomic_add_fetch_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	_, new := doAtomicAdd8(ptr, value)
	return new
}

//go:inline
func doAtomicSub8(ptr *uint8, value uint8) (old, new uint8) {
	mask := interrupt.Disable()
	old = *ptr
	new = old - value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_sub_1
func __atomic_fetch_sub_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	old, _ := doAtomicSub8(ptr, value)
	return old
}

//export __sync_fetch_and_sub_1
func __sync_fetch_and_sub_1(ptr *uint8, value uint8) uint8 {
	old, _ := doAtomicSub8(ptr, value)
	return old
}

//export __atomic_sub_fetch_1
func __atomic_sub_fetch_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	_, new := doAtomicSub8(ptr, value)
	return new
}

//go:inline
func doAtomicAnd8(ptr *uint8, value uint8) (old, new uint8) {
	mask := interrupt.Disable()
	old = *ptr
	new = old & value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_and_1
func __atomic_fetch_and_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	old, _ := doAtomicAnd8(ptr, value)
	return old
}

//export __sync_fetch_and_and_1
func __sync_fetch_and_and_1(ptr *uint8, value uint8) uint8 {
	old, _ := doAtomicAnd8(ptr, value)
	return old
}

//export __atomic_and_fetch_1
func __atomic_and_fetch_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	_, new := doAtomicAnd8(ptr, value)
	return new
}

//go:inline
func doAtomicOr8(ptr *uint8, value uint8) (old, new uint8) {
	mask := interrupt.Disable()
	old = *ptr
	new = old | value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_or_1
func __atomic_fetch_or_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	old, _ := doAtomicOr8(ptr, value)
	return old
}

//export __sync_fetch_and_or_1
func __sync_fetch_and_or_1(ptr *uint8, value uint8) uint8 {
	old, _ := doAtomicOr8(ptr, value)
	return old
}

//export __atomic_or_fetch_1
func __atomic_or_fetch_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	_, new := doAtomicOr8(ptr, value)
	return new
}

//go:inline
func doAtomicXor8(ptr *uint8, value uint8) (old, new uint8) {
	mask := interrupt.Disable()
	old = *ptr
	new = old ^ value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_xor_1
func __atomic_fetch_xor_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	old, _ := doAtomicXor8(ptr, value)
	return old
}

//export __sync_fetch_and_xor_1
func __sync_fetch_and_xor_1(ptr *uint8, value uint8) uint8 {
	old, _ := doAtomicXor8(ptr, value)
	return old
}

//export __atomic_xor_fetch_1
func __atomic_xor_fetch_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	_, new := doAtomicXor8(ptr, value)
	return new
}

//go:inline
func doAtomicNand8(ptr *uint8, value uint8) (old, new uint8) {
	mask := interrupt.Disable()
	old = *ptr
	new = ^(old & value)
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_nand_1
func __atomic_fetch_nand_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	old, _ := doAtomicNand8(ptr, value)
	return old
}

//export __sync_fetch_and_nand_1
func __sync_fetch_and_nand_1(ptr *uint8, value uint8) uint8 {
	old, _ := doAtomicNand8(ptr, value)
	return old
}

//export __atomic_nand_fetch_1
func __atomic_nand_fetch_1(ptr *uint8, value uint8, ordering uintptr) uint8 {
	_, new := doAtomicNand8(ptr, value)
	return new
}

// 16-bit atomics.

//export __atomic_load_2
func __atomic_load_2(ptr *uint16, ordering uintptr) uint16 {
	// The LLVM docs for this have a typo saying that there is a val argument after the pointer.
	// That is a typo, and the GCC docs omit it.
	mask := interrupt.Disable()
	val := *ptr
	interrupt.Restore(mask)
	return val
}

//export __atomic_store_2
func __atomic_store_2(ptr *uint16, val uint16, ordering uintptr) {
	mask := interrupt.Disable()
	*ptr = val
	interrupt.Restore(mask)
}

//go:inline
func doAtomicCAS16(ptr *uint16, expected, desired uint16) uint16 {
	mask := interrupt.Disable()
	old := *ptr
	if old == expected {
		*ptr = desired
	}
	interrupt.Restore(mask)
	return old
}

//export __sync_val_compare_and_swap_2
func __sync_val_compare_and_swap_2(ptr *uint16, expected, desired uint16) uint16 {
	return doAtomicCAS16(ptr, expected, desired)
}

//export __atomic_compare_exchange_2
func __atomic_compare_exchange_2(ptr, expected *uint16, desired uint16, successOrder, failureOrder uintptr) bool {
	exp := *expected
	old := doAtomicCAS16(ptr, exp, desired)
	return old == exp
}

//go:inline
func doAtomicSwap16(ptr *uint16, new uint16) uint16 {
	mask := interrupt.Disable()
	old := *ptr
	*ptr = new
	interrupt.Restore(mask)
	return old
}

//export __sync_lock_test_and_set_2
func __sync_lock_test_and_set_2(ptr *uint16, new uint16) uint16 {
	return doAtomicSwap16(ptr, new)
}

//export __atomic_exchange_2
func __atomic_exchange_2(ptr *uint16, new uint16, ordering uintptr) uint16 {
	return doAtomicSwap16(ptr, new)
}

//go:inline
func doAtomicAdd16(ptr *uint16, value uint16) (old, new uint16) {
	mask := interrupt.Disable()
	old = *ptr
	new = old + value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_add_2
func __atomic_fetch_add_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	old, _ := doAtomicAdd16(ptr, value)
	return old
}

//export __sync_fetch_and_add_2
func __sync_fetch_and_add_2(ptr *uint16, value uint16) uint16 {
	old, _ := doAtomicAdd16(ptr, value)
	return old
}

//export __atomic_add_fetch_2
func __atomic_add_fetch_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	_, new := doAtomicAdd16(ptr, value)
	return new
}

//go:inline
func doAtomicSub16(ptr *uint16, value uint16) (old, new uint16) {
	mask := interrupt.Disable()
	old = *ptr
	new = old - value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_sub_2
func __atomic_fetch_sub_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	old, _ := doAtomicSub16(ptr, value)
	return old
}

//export __sync_fetch_and_sub_2
func __sync_fetch_and_sub_2(ptr *uint16, value uint16) uint16 {
	old, _ := doAtomicSub16(ptr, value)
	return old
}

//export __atomic_sub_fetch_2
func __atomic_sub_fetch_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	_, new := doAtomicSub16(ptr, value)
	return new
}

//go:inline
func doAtomicAnd16(ptr *uint16, value uint16) (old, new uint16) {
	mask := interrupt.Disable()
	old = *ptr
	new = old & value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_and_2
func __atomic_fetch_and_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	old, _ := doAtomicAnd16(ptr, value)
	return old
}

//export __sync_fetch_and_and_2
func __sync_fetch_and_and_2(ptr *uint16, value uint16) uint16 {
	old, _ := doAtomicAnd16(ptr, value)
	return old
}

//export __atomic_and_fetch_2
func __atomic_and_fetch_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	_, new := doAtomicAnd16(ptr, value)
	return new
}

//go:inline
func doAtomicOr16(ptr *uint16, value uint16) (old, new uint16) {
	mask := interrupt.Disable()
	old = *ptr
	new = old | value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_or_2
func __atomic_fetch_or_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	old, _ := doAtomicOr16(ptr, value)
	return old
}

//export __sync_fetch_and_or_2
func __sync_fetch_and_or_2(ptr *uint16, value uint16) uint16 {
	old, _ := doAtomicOr16(ptr, value)
	return old
}

//export __atomic_or_fetch_2
func __atomic_or_fetch_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	_, new := doAtomicOr16(ptr, value)
	return new
}

//go:inline
func doAtomicXor16(ptr *uint16, value uint16) (old, new uint16) {
	mask := interrupt.Disable()
	old = *ptr
	new = old ^ value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_xor_2
func __atomic_fetch_xor_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	old, _ := doAtomicXor16(ptr, value)
	return old
}

//export __sync_fetch_and_xor_2
func __sync_fetch_and_xor_2(ptr *uint16, value uint16) uint16 {
	old, _ := doAtomicXor16(ptr, value)
	return old
}

//export __atomic_xor_fetch_2
func __atomic_xor_fetch_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	_, new := doAtomicXor16(ptr, value)
	return new
}

//go:inline
func doAtomicNand16(ptr *uint16, value uint16) (old, new uint16) {
	mask := interrupt.Disable()
	old = *ptr
	new = ^(old & value)
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_nand_2
func __atomic_fetch_nand_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	old, _ := doAtomicNand16(ptr, value)
	return old
}

//export __sync_fetch_and_nand_2
func __sync_fetch_and_nand_2(ptr *uint16, value uint16) uint16 {
	old, _ := doAtomicNand16(ptr, value)
	return old
}

//export __atomic_nand_fetch_2
func __atomic_nand_fetch_2(ptr *uint16, value uint16, ordering uintptr) uint16 {
	_, new := doAtomicNand16(ptr, value)
	return new
}

// 32-bit atomics.

//export __atomic_load_4
func __atomic_load_4(ptr *uint32, ordering uintptr) uint32 {
	// The LLVM docs for this have a typo saying that there is a val argument after the pointer.
	// That is a typo, and the GCC docs omit it.
	mask := interrupt.Disable()
	val := *ptr
	interrupt.Restore(mask)
	return val
}

//export __atomic_store_4
func __atomic_store_4(ptr *uint32, val uint32, ordering uintptr) {
	mask := interrupt.Disable()
	*ptr = val
	interrupt.Restore(mask)
}

//go:inline
func doAtomicCAS32(ptr *uint32, expected, desired uint32) uint32 {
	mask := interrupt.Disable()
	old := *ptr
	if old == expected {
		*ptr = desired
	}
	interrupt.Restore(mask)
	return old
}

//export __sync_val_compare_and_swap_4
func __sync_val_compare_and_swap_4(ptr *uint32, expected, desired uint32) uint32 {
	return doAtomicCAS32(ptr, expected, desired)
}

//export __atomic_compare_exchange_4
func __atomic_compare_exchange_4(ptr, expected *uint32, desired uint32, successOrder, failureOrder uintptr) bool {
	exp := *expected
	old := doAtomicCAS32(ptr, exp, desired)
	return old == exp
}

//go:inline
func doAtomicSwap32(ptr *uint32, new uint32) uint32 {
	mask := interrupt.Disable()
	old := *ptr
	*ptr = new
	interrupt.Restore(mask)
	return old
}

//export __sync_lock_test_and_set_4
func __sync_lock_test_and_set_4(ptr *uint32, new uint32) uint32 {
	return doAtomicSwap32(ptr, new)
}

//export __atomic_exchange_4
func __atomic_exchange_4(ptr *uint32, new uint32, ordering uintptr) uint32 {
	return doAtomicSwap32(ptr, new)
}

//go:inline
func doAtomicAdd32(ptr *uint32, value uint32) (old, new uint32) {
	mask := interrupt.Disable()
	old = *ptr
	new = old + value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_add_4
func __atomic_fetch_add_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	old, _ := doAtomicAdd32(ptr, value)
	return old
}

//export __sync_fetch_and_add_4
func __sync_fetch_and_add_4(ptr *uint32, value uint32) uint32 {
	old, _ := doAtomicAdd32(ptr, value)
	return old
}

//export __atomic_add_fetch_4
func __atomic_add_fetch_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	_, new := doAtomicAdd32(ptr, value)
	return new
}

//go:inline
func doAtomicSub32(ptr *uint32, value uint32) (old, new uint32) {
	mask := interrupt.Disable()
	old = *ptr
	new = old - value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_sub_4
func __atomic_fetch_sub_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	old, _ := doAtomicSub32(ptr, value)
	return old
}

//export __sync_fetch_and_sub_4
func __sync_fetch_and_sub_4(ptr *uint32, value uint32) uint32 {
	old, _ := doAtomicSub32(ptr, value)
	return old
}

//export __atomic_sub_fetch_4
func __atomic_sub_fetch_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	_, new := doAtomicSub32(ptr, value)
	return new
}

//go:inline
func doAtomicAnd32(ptr *uint32, value uint32) (old, new uint32) {
	mask := interrupt.Disable()
	old = *ptr
	new = old & value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_and_4
func __atomic_fetch_and_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	old, _ := doAtomicAnd32(ptr, value)
	return old
}

//export __sync_fetch_and_and_4
func __sync_fetch_and_and_4(ptr *uint32, value uint32) uint32 {
	old, _ := doAtomicAnd32(ptr, value)
	return old
}

//export __atomic_and_fetch_4
func __atomic_and_fetch_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	_, new := doAtomicAnd32(ptr, value)
	return new
}

//go:inline
func doAtomicOr32(ptr *uint32, value uint32) (old, new uint32) {
	mask := interrupt.Disable()
	old = *ptr
	new = old | value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_or_4
func __atomic_fetch_or_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	old, _ := doAtomicOr32(ptr, value)
	return old
}

//export __sync_fetch_and_or_4
func __sync_fetch_and_or_4(ptr *uint32, value uint32) uint32 {
	old, _ := doAtomicOr32(ptr, value)
	return old
}

//export __atomic_or_fetch_4
func __atomic_or_fetch_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	_, new := doAtomicOr32(ptr, value)
	return new
}

//go:inline
func doAtomicXor32(ptr *uint32, value uint32) (old, new uint32) {
	mask := interrupt.Disable()
	old = *ptr
	new = old ^ value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_xor_4
func __atomic_fetch_xor_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	old, _ := doAtomicXor32(ptr, value)
	return old
}

//export __sync_fetch_and_xor_4
func __sync_fetch_and_xor_4(ptr *uint32, value uint32) uint32 {
	old, _ := doAtomicXor32(ptr, value)
	return old
}

//export __atomic_xor_fetch_4
func __atomic_xor_fetch_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	_, new := doAtomicXor32(ptr, value)
	return new
}

//go:inline
func doAtomicNand32(ptr *uint32, value uint32) (old, new uint32) {
	mask := interrupt.Disable()
	old = *ptr
	new = ^(old & value)
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_nand_4
func __atomic_fetch_nand_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	old, _ := doAtomicNand32(ptr, value)
	return old
}

//export __sync_fetch_and_nand_4
func __sync_fetch_and_nand_4(ptr *uint32, value uint32) uint32 {
	old, _ := doAtomicNand32(ptr, value)
	return old
}

//export __atomic_nand_fetch_4
func __atomic_nand_fetch_4(ptr *uint32, value uint32, ordering uintptr) uint32 {
	_, new := doAtomicNand32(ptr, value)
	return new
}

// 64-bit atomics.

//export __atomic_load_8
func __atomic_load_8(ptr *uint64, ordering uintptr) uint64 {
	// The LLVM docs for this have a typo saying that there is a val argument after the pointer.
	// That is a typo, and the GCC docs omit it.
	mask := interrupt.Disable()
	val := *ptr
	interrupt.Restore(mask)
	return val
}

//export __atomic_store_8
func __atomic_store_8(ptr *uint64, val uint64, ordering uintptr) {
	mask := interrupt.Disable()
	*ptr = val
	interrupt.Restore(mask)
}

//go:inline
func doAtomicCAS64(ptr *uint64, expected, desired uint64) uint64 {
	mask := interrupt.Disable()
	old := *ptr
	if old == expected {
		*ptr = desired
	}
	interrupt.Restore(mask)
	return old
}

//export __sync_val_compare_and_swap_8
func __sync_val_compare_and_swap_8(ptr *uint64, expected, desired uint64) uint64 {
	return doAtomicCAS64(ptr, expected, desired)
}

//export __atomic_compare_exchange_8
func __atomic_compare_exchange_8(ptr, expected *uint64, desired uint64, successOrder, failureOrder uintptr) bool {
	exp := *expected
	old := doAtomicCAS64(ptr, exp, desired)
	return old == exp
}

//go:inline
func doAtomicSwap64(ptr *uint64, new uint64) uint64 {
	mask := interrupt.Disable()
	old := *ptr
	*ptr = new
	interrupt.Restore(mask)
	return old
}

//export __sync_lock_test_and_set_8
func __sync_lock_test_and_set_8(ptr *uint64, new uint64) uint64 {
	return doAtomicSwap64(ptr, new)
}

//export __atomic_exchange_8
func __atomic_exchange_8(ptr *uint64, new uint64, ordering uintptr) uint64 {
	return doAtomicSwap64(ptr, new)
}

//go:inline
func doAtomicAdd64(ptr *uint64, value uint64) (old, new uint64) {
	mask := interrupt.Disable()
	old = *ptr
	new = old + value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_add_8
func __atomic_fetch_add_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	old, _ := doAtomicAdd64(ptr, value)
	return old
}

//export __sync_fetch_and_add_8
func __sync_fetch_and_add_8(ptr *uint64, value uint64) uint64 {
	old, _ := doAtomicAdd64(ptr, value)
	return old
}

//export __atomic_add_fetch_8
func __atomic_add_fetch_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	_, new := doAtomicAdd64(ptr, value)
	return new
}

//go:inline
func doAtomicSub64(ptr *uint64, value uint64) (old, new uint64) {
	mask := interrupt.Disable()
	old = *ptr
	new = old - value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_sub_8
func __atomic_fetch_sub_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	old, _ := doAtomicSub64(ptr, value)
	return old
}

//export __sync_fetch_and_sub_8
func __sync_fetch_and_sub_8(ptr *uint64, value uint64) uint64 {
	old, _ := doAtomicSub64(ptr, value)
	return old
}

//export __atomic_sub_fetch_8
func __atomic_sub_fetch_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	_, new := doAtomicSub64(ptr, value)
	return new
}

//go:inline
func doAtomicAnd64(ptr *uint64, value uint64) (old, new uint64) {
	mask := interrupt.Disable()
	old = *ptr
	new = old & value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_and_8
func __atomic_fetch_and_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	old, _ := doAtomicAnd64(ptr, value)
	return old
}

//export __sync_fetch_and_and_8
func __sync_fetch_and_and_8(ptr *uint64, value uint64) uint64 {
	old, _ := doAtomicAnd64(ptr, value)
	return old
}

//export __atomic_and_fetch_8
func __atomic_and_fetch_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	_, new := doAtomicAnd64(ptr, value)
	return new
}

//go:inline
func doAtomicOr64(ptr *uint64, value uint64) (old, new uint64) {
	mask := interrupt.Disable()
	old = *ptr
	new = old | value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_or_8
func __atomic_fetch_or_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	old, _ := doAtomicOr64(ptr, value)
	return old
}

//export __sync_fetch_and_or_8
func __sync_fetch_and_or_8(ptr *uint64, value uint64) uint64 {
	old, _ := doAtomicOr64(ptr, value)
	return old
}

//export __atomic_or_fetch_8
func __atomic_or_fetch_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	_, new := doAtomicOr64(ptr, value)
	return new
}

//go:inline
func doAtomicXor64(ptr *uint64, value uint64) (old, new uint64) {
	mask := interrupt.Disable()
	old = *ptr
	new = old ^ value
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_xor_8
func __atomic_fetch_xor_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	old, _ := doAtomicXor64(ptr, value)
	return old
}

//export __sync_fetch_and_xor_8
func __sync_fetch_and_xor_8(ptr *uint64, value uint64) uint64 {
	old, _ := doAtomicXor64(ptr, value)
	return old
}

//export __atomic_xor_fetch_8
func __atomic_xor_fetch_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	_, new := doAtomicXor64(ptr, value)
	return new
}

//go:inline
func doAtomicNand64(ptr *uint64, value uint64) (old, new uint64) {
	mask := interrupt.Disable()
	old = *ptr
	new = ^(old & value)
	*ptr = new
	interrupt.Restore(mask)
	return old, new
}

//export __atomic_fetch_nand_8
func __atomic_fetch_nand_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	old, _ := doAtomicNand64(ptr, value)
	return old
}

//export __sync_fetch_and_nand_8
func __sync_fetch_and_nand_8(ptr *uint64, value uint64) uint64 {
	old, _ := doAtomicNand64(ptr, value)
	return old
}

//export __atomic_nand_fetch_8
func __atomic_nand_fetch_8(ptr *uint64, value uint64, ordering uintptr) uint64 {
	_, new := doAtomicNand64(ptr, value)
	return new
}
