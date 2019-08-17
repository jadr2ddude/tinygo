target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7m-none-eabi"

%runtime.hashmap = type { %runtime.hashmap*, i8*, i32, i8, i8, i8 }

declare nonnull %runtime.hashmap* @runtime.hashmapMake(i8 %keySize, i8 %valueSize, i32 %sizeHint)

define void @testUnused() {
    %1 = call %runtime.hashmap* @runtime.hashmapMake(i8 4, i8 4, i32 0)
    ret void
}

define %runtime.hashmap* @testUsed() {
    %1 = call %runtime.hashmap* @runtime.hashmapMake(i8 4, i8 4, i32 0)
    ret %runtime.hashmap* %1
}
