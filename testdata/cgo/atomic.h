#include <stdbool.h>

int atomicLoad();
void atomicStore(int v);
int atomicSwap(int v);
bool atomicCAS(int old, int new);
int atomicAdd(int v);
int atomicSub(int v);
int atomicOr(int v);
int atomicXOr(int v);
int atomicAnd(int v);


char atomicCharLoad();
char atomicCharAdd(char v);