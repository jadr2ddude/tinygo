target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7m-none-eabi"

%chain = type { %chain* }

declare void @foo()
declare i8 @usePtr(i8*)
declare i1 @next(i8*)

; nothing does nothing.
; The soft-stack pass should ignore it.
define void @nothing() {
entry:
    ret void
}

; callNothing calls "nothing".
; The soft-stack pass should also ignore it.
define void @callNothing() {
entry:
    call void @nothing()
    ret void
}

; wrap calls a potentially-blocking function and immediately returns.
; TODO: this actually does not require transformation, we can maybe add a special case for simple wrappers
define void @wrap() {
entry:
    call void @foo()
    ret void
}

; frameAlloca contains an alloca which must be transferred to the soft-stack frame.
define i8 @frameAlloca() {
entry:
    %ptr = alloca i8
    %ret = call i8 @usePtr(i8* %ptr)
    ret i8 %ret
}

; nonStaticAlloca contains a non-static alloca which must be executed on the soft-stack.
define i8 @nonStaticAlloca(i32 %len) {
entry:
    br label %loop

loop:
    %n = phi i32 [ %len , %entry ], [ %n.next, %loop  ]
    %prev = phi %chain* [ null, %entry ], [ %ptr, %loop ]
    %ptr = alloca %chain
    %elem = getelementptr inbounds %chain, %chain* %ptr, i32 0, i32 0
    store %chain* %prev, %chain** %elem
    %n.next = sub i32 %n, 1
    %done = icmp eq i32 %n.next, 0
    br i1 %done, label %end, label %loop

end:
    %ptr.cast = bitcast %chain* %ptr to i8*
    %ret = call i8 @usePtr(i8* %ptr.cast)
    ret i8 %ret
}

; multiCall calls multiple blocking functions.
define i8 @multiCall(i8* %ptr) {
entry:
    %val = call i8 @usePtr(i8* %ptr)
    call void @foo()
    ret i8 %val
}

; savelessLoop is a "for x.next()"-style loop with a non-blocking body.
; This does not require *any* saves (including control flow).
; As such, nothing will actually be placed on the soft-stack, but the unwind chain will need to be handled.
define void @savelessLoop(i8* %x) {
entry:
    br label %loop.header

loop.header:
    %next = call i1 @next(i8* %x)
    br i1 %next, label %loop.body, label %end

loop.body:
    call void @callNothing()
    br label %loop.header

end:
    ret void
}

define void @icall(i8* (i8*, i8*)* %fn.ptr, i8* %fn.ctx, i8* %start) {
entry:
    br label %loop

loop:
    %ptr = phi i8* [ %start, %entry ], [ %next, %loop ]
    %next = call i8* %fn.ptr(i8* %fn.ctx, i8* %ptr)
    %done = icmp eq i8* %next, null
    br i1 %done, label %end, label %loop

end:
    ret void
}
