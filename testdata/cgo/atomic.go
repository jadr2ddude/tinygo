package main

// #include "atomic.h"
// #cgo CFLAGS: -std=c11
import "C"

func testAtomics() {
	println("test atomics")
	println("load initializer:", C.atomicLoad())
	C.atomicStore(5)
	println("load after storing 5:", C.atomicLoad())
	println("previous value from swap to 7:", C.atomicSwap(7))
	println("load after swap:", C.atomicLoad())
	println("atomic cas fail:", C.atomicCAS(5, 6))
	println("load after failed cas:", C.atomicLoad())
	println("atomic cas swap:", C.atomicCAS(7, 6))
	println("load after cas swap:", C.atomicLoad())
	println("previous value from add:", C.atomicAdd(2))
	println("previous value from sub:", C.atomicSub(5))
	println("previous value from or:", C.atomicOr(0b101))
	println("previous value from xor:", C.atomicXOr(0b010))
	println("previous value from and:", C.atomicAnd(0b011))
	println("final value:", C.atomicLoad())
	println("atomic char load:", C.atomicCharLoad())
	println("previous value from char add:", C.atomicCharAdd(1))
	println("atomic char load after add:", C.atomicCharLoad())
}
