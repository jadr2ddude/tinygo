//go:build scheduler.coroutines || scheduler.softstack
// +build scheduler.coroutines scheduler.softstack

package runtime

// getSystemStackPointer returns the current stack pointer of the system stack.
// This is always the current stack pointer.
func getSystemStackPointer() uintptr {
	return getCurrentStackPointer()
}
