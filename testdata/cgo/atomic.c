#include <stdbool.h>
#include <stdatomic.h>

#pragma clang diagnostic ignored "-Watomic-alignment"

static volatile atomic_int counter = 0;

int atomicLoad() {
    return atomic_load(&counter);
}

void atomicStore(int v) {
    return atomic_store(&counter, v);
}

int atomicSwap(int v) {
    return atomic_exchange(&counter, v);
}

bool atomicCAS(int old, int new) {
    return atomic_compare_exchange_strong(&counter, &old, new);
}

int atomicAdd(int v) {
    return atomic_fetch_add(&counter, v);
}

int atomicSub(int v) {
    return atomic_fetch_sub(&counter, v);
}

int atomicOr(int v) {
    return atomic_fetch_or(&counter, v);
}

int atomicXOr(int v) {
    return atomic_fetch_xor(&counter, v);
}

int atomicAnd(int v) {
    return atomic_fetch_and(&counter, v);
}

static volatile atomic_char achar = 'x';

char atomicCharLoad() {
    return atomic_load(&achar);
}

char atomicCharAdd(char v) {
    return atomic_fetch_add(&achar, v);
}
