target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7m-none-eabi"

%chain = type { %chain* }

@"(internal/task).windchain" = external global i8*
@"(internal/task).sp" = external global i8*

declare void @foo()

declare i8 @usePtr(i8*)

declare i1 @next(i8*)

define void @nothing() {
entry:
  ret void
}

define void @callNothing() {
entry:
  call void @nothing()
  ret void
}

define void @wrap() {
entry:
  %rewind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %rewind.check = icmp ne i8* %rewind.ptr, null
  br i1 %rewind.check, label %entry.call, label %entry.normal

entry.normal:                                     ; preds = %entry
  br label %entry.call

entry.call:                                       ; preds = %entry, %entry.normal
  call void @foo()
  %unwind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %unwind.check = icmp ne i8* %unwind.ptr, null
  br i1 %unwind.check, label %unwind, label %entry.call.after

unwind:                                           ; preds = %entry.call
  ret void

entry.call.after:                                 ; preds = %entry.call, <null operand!>
  ret void
}

define i8 @frameAlloca() {
entry:
  %frame.raw = alloca <{ i8 }>, align 4
  %rewind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %rewind.check = icmp ne i8* %rewind.ptr, null
  br i1 %rewind.check, label %entry.entry.call.rewind, label %entry.normal

entry.normal:                                     ; preds = %entry
  %entry.sp = load i8*, i8** @"(internal/task).sp", align 4
  %entry.onSystemStack = icmp eq i8* %entry.sp, null
  br i1 %entry.onSystemStack, label %entry.raw, label %entry.soft

entry.soft:                                       ; preds = %entry.normal
  %frame.soft.sp = load i8*, i8** @"(internal/task).sp", align 4
  %frame.soft.sp.int = ptrtoint i8* %frame.soft.sp to i32
  %frame.soft.sp.sub = sub i32 %frame.soft.sp.int, 1
  %frame.soft.sp.new = inttoptr i32 %frame.soft.sp.sub to i8*
  store i8* %frame.soft.sp.new, i8** @"(internal/task).sp", align 4
  %frame.soft = bitcast i8* %frame.soft.sp.new to <{ i8 }>*
  br label %entry.entry

entry.raw:                                        ; preds = %entry.normal
  %frame.raw.bitcast = bitcast <{ i8 }>* %frame.raw to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %frame.raw.bitcast)
  br label %entry.entry

entry.entry:                                      ; preds = %entry.soft, %entry.raw
  %frame.ptr = phi <{ i8 }>* [ %frame.raw, %entry.raw ], [ %frame.soft, %entry.soft ]
  %ptr.ptr = getelementptr inbounds <{ i8 }>, <{ i8 }>* %frame.ptr, i32 0, i32 0
  br label %entry.entry.call

entry.entry.call.rewind:                          ; preds = %entry
  %rewind.next.ptr = bitcast i8* %rewind.ptr to i8**
  %rewind.next = load i8*, i8** %rewind.next.ptr, align 4
  store i8* %rewind.next, i8** @"(internal/task).windchain", align 4
  %entry.entry.call.rewind.superframe = bitcast i8* %rewind.ptr to { i8*, { i8*, i8* } }*
  %entry.entry.call.rewind.frame = getelementptr inbounds { i8*, { i8*, i8* } }, { i8*, { i8*, i8* } }* %entry.entry.call.rewind.superframe, i32 0, i32 1
  %ptr.ptr.save.0.reload.ptr = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %entry.entry.call.rewind.frame, i32 0, i32 0
  %ptr.ptr.save.0.reload = load i8*, i8** %ptr.ptr.save.0.reload.ptr, align 4
  %entry.sp.save.0.reload.ptr = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %entry.entry.call.rewind.frame, i32 0, i32 1
  %entry.sp.save.0.reload = load i8*, i8** %entry.sp.save.0.reload.ptr, align 4
  br label %entry.entry.call

entry.entry.call:                                 ; preds = %entry.entry.call.rewind, %entry.entry
  %entry.sp.save.0 = phi i8* [ %entry.sp, %entry.entry ], [ %entry.sp.save.0.reload, %entry.entry.call.rewind ]
  %ptr.ptr.save.0 = phi i8* [ %ptr.ptr, %entry.entry ], [ %ptr.ptr.save.0.reload, %entry.entry.call.rewind ]
  %ret = call i8 @usePtr(i8* %ptr.ptr.save.0)
  %entry.entry.call.unwind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %entry.entry.call.unwind.check = icmp ne i8* %entry.entry.call.unwind.ptr, null
  br i1 %entry.entry.call.unwind.check, label %entry.entry.call.unwind, label %entry.entry.call.after

entry.entry.call.unwind:                          ; preds = %entry.entry.call
  %entry.entry.call.unwind.frame.sp = load i8*, i8** @"(internal/task).sp", align 4
  %entry.entry.call.unwind.frame.sp.int = ptrtoint i8* %entry.entry.call.unwind.frame.sp to i32
  %entry.entry.call.unwind.frame.sp.sub = sub i32 %entry.entry.call.unwind.frame.sp.int, 12
  %entry.entry.call.unwind.frame.sp.align = and i32 %entry.entry.call.unwind.frame.sp.sub, -4
  %entry.entry.call.unwind.frame.sp.new = inttoptr i32 %entry.entry.call.unwind.frame.sp.align to i8*
  store i8* %entry.entry.call.unwind.frame.sp.new, i8** @"(internal/task).sp", align 4
  %entry.entry.call.unwind.frame = bitcast i8* %entry.entry.call.unwind.frame.sp.new to { i8*, { i8*, i8* } }*
  %entry.entry.call.unwind.frame.prev = getelementptr inbounds { i8*, { i8*, i8* } }, { i8*, { i8*, i8* } }* %entry.entry.call.unwind.frame, i32 0, i32 0
  store i8* %entry.entry.call.unwind.ptr, i8** %entry.entry.call.unwind.frame.prev, align 4
  %entry.entry.call.unwind.frame.frame = getelementptr inbounds { i8*, { i8*, i8* } }, { i8*, { i8*, i8* } }* %entry.entry.call.unwind.frame, i32 0, i32 1
  %ptr.ptr.save.0.ptr = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %entry.entry.call.unwind.frame.frame, i32 0, i32 0
  store i8* %ptr.ptr.save.0, i8** %ptr.ptr.save.0.ptr, align 4
  %entry.sp.save.0.ptr = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %entry.entry.call.unwind.frame.frame, i32 0, i32 1
  store i8* %entry.sp.save.0, i8** %entry.sp.save.0.ptr, align 4
  %entry.entry.call.unwind.frame.cast = bitcast { i8*, { i8*, i8* } }* %entry.entry.call.unwind.frame to i8*
  store i8* %entry.entry.call.unwind.frame.cast, i8** @"(internal/task).windchain", align 4
  ret i8 undef

entry.entry.call.after:                           ; preds = %entry.entry.call, <null operand!>
  store i8* %entry.sp.save.0, i8** @"(internal/task).sp", align 4
  ret i8 %ret
}

define i8 @nonStaticAlloca(i32 %len) {
entry:
  %frame.raw = alloca <{}>, align 4
  %rewind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %rewind.check = icmp ne i8* %rewind.ptr, null
  br i1 %rewind.check, label %end.call.rewind, label %entry.normal

entry.normal:                                     ; preds = %entry
  %entry.sp = load i8*, i8** @"(internal/task).sp", align 4
  %entry.onSystemStack = icmp eq i8* %entry.sp, null
  br i1 %entry.onSystemStack, label %entry.raw, label %entry.soft

entry.soft:                                       ; preds = %entry.normal
  %frame.soft.sp = load i8*, i8** @"(internal/task).sp", align 4
  %frame.soft.sp.int = ptrtoint i8* %frame.soft.sp to i32
  %frame.soft.sp.sub = sub i32 %frame.soft.sp.int, 0
  %frame.soft.sp.new = inttoptr i32 %frame.soft.sp.sub to i8*
  store i8* %frame.soft.sp.new, i8** @"(internal/task).sp", align 4
  %frame.soft = bitcast i8* %frame.soft.sp.new to <{}>*
  br label %entry.entry

entry.raw:                                        ; preds = %entry.normal
  %frame.raw.bitcast = bitcast <{}>* %frame.raw to i8*
  call void @llvm.lifetime.start.p0i8(i64 0, i8* %frame.raw.bitcast)
  br label %entry.entry

entry.entry:                                      ; preds = %entry.soft, %entry.raw
  %frame.ptr = phi <{}>* [ %frame.raw, %entry.raw ], [ %frame.soft, %entry.soft ]
  br label %loop

loop.continued:                                   ; preds = %ptr.soft.block, %ptr.raw.block
  %ptr = phi %chain* [ %0, %ptr.raw.block ], [ %ptr.soft, %ptr.soft.block ]
  %elem = getelementptr inbounds %chain, %chain* %ptr, i32 0, i32 0
  store %chain* %prev, %chain** %elem, align 4
  %n.next = sub i32 %n, 1
  %done = icmp eq i32 %n.next, 0
  br i1 %done, label %end, label %loop

loop:                                             ; preds = %loop.continued, %entry.entry
  %n = phi i32 [ %len, %entry.entry ], [ %n.next, %loop.continued ]
  %prev = phi %chain* [ null, %entry.entry ], [ %ptr, %loop.continued ]
  br i1 %entry.onSystemStack, label %ptr.raw.block, label %ptr.soft.block

ptr.soft.block:                                   ; preds = %loop
  %ptr.soft.sp = load i8*, i8** @"(internal/task).sp", align 4
  %ptr.soft.sp.int = ptrtoint i8* %ptr.soft.sp to i32
  %ptr.soft.sp.sub = sub i32 %ptr.soft.sp.int, 4
  %ptr.soft.sp.align = and i32 %ptr.soft.sp.sub, -4
  %ptr.soft.sp.new = inttoptr i32 %ptr.soft.sp.align to i8*
  store i8* %ptr.soft.sp.new, i8** @"(internal/task).sp", align 4
  %ptr.soft = bitcast i8* %ptr.soft.sp.new to %chain*
  br label %loop.continued

ptr.raw.block:                                    ; preds = %loop
  %0 = alloca %chain, align 4
  br label %loop.continued

end:                                              ; preds = %loop.continued
  %ptr.cast = bitcast %chain* %ptr to i8*
  br label %end.call

end.call.rewind:                                  ; preds = %entry
  %rewind.next.ptr = bitcast i8* %rewind.ptr to i8**
  %rewind.next = load i8*, i8** %rewind.next.ptr, align 4
  store i8* %rewind.next, i8** @"(internal/task).windchain", align 4
  %end.call.rewind.superframe = bitcast i8* %rewind.ptr to { i8*, { i8*, i8* } }*
  %end.call.rewind.frame = getelementptr inbounds { i8*, { i8*, i8* } }, { i8*, { i8*, i8* } }* %end.call.rewind.superframe, i32 0, i32 1
  %ptr.cast.save.0.reload.ptr = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %end.call.rewind.frame, i32 0, i32 0
  %ptr.cast.save.0.reload = load i8*, i8** %ptr.cast.save.0.reload.ptr, align 4
  %entry.sp.save.0.reload.ptr = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %end.call.rewind.frame, i32 0, i32 1
  %entry.sp.save.0.reload = load i8*, i8** %entry.sp.save.0.reload.ptr, align 4
  br label %end.call

end.call:                                         ; preds = %end.call.rewind, %end
  %entry.sp.save.0 = phi i8* [ %entry.sp, %end ], [ %entry.sp.save.0.reload, %end.call.rewind ]
  %ptr.cast.save.0 = phi i8* [ %ptr.cast, %end ], [ %ptr.cast.save.0.reload, %end.call.rewind ]
  %ret = call i8 @usePtr(i8* %ptr.cast.save.0)
  %end.call.unwind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %end.call.unwind.check = icmp ne i8* %end.call.unwind.ptr, null
  br i1 %end.call.unwind.check, label %end.call.unwind, label %end.call.after

end.call.unwind:                                  ; preds = %end.call
  %end.call.unwind.frame.sp = load i8*, i8** @"(internal/task).sp", align 4
  %end.call.unwind.frame.sp.int = ptrtoint i8* %end.call.unwind.frame.sp to i32
  %end.call.unwind.frame.sp.sub = sub i32 %end.call.unwind.frame.sp.int, 12
  %end.call.unwind.frame.sp.align = and i32 %end.call.unwind.frame.sp.sub, -4
  %end.call.unwind.frame.sp.new = inttoptr i32 %end.call.unwind.frame.sp.align to i8*
  store i8* %end.call.unwind.frame.sp.new, i8** @"(internal/task).sp", align 4
  %end.call.unwind.frame = bitcast i8* %end.call.unwind.frame.sp.new to { i8*, { i8*, i8* } }*
  %end.call.unwind.frame.prev = getelementptr inbounds { i8*, { i8*, i8* } }, { i8*, { i8*, i8* } }* %end.call.unwind.frame, i32 0, i32 0
  store i8* %end.call.unwind.ptr, i8** %end.call.unwind.frame.prev, align 4
  %end.call.unwind.frame.frame = getelementptr inbounds { i8*, { i8*, i8* } }, { i8*, { i8*, i8* } }* %end.call.unwind.frame, i32 0, i32 1
  %ptr.cast.save.0.ptr = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %end.call.unwind.frame.frame, i32 0, i32 0
  store i8* %ptr.cast.save.0, i8** %ptr.cast.save.0.ptr, align 4
  %entry.sp.save.0.ptr = getelementptr inbounds { i8*, i8* }, { i8*, i8* }* %end.call.unwind.frame.frame, i32 0, i32 1
  store i8* %entry.sp.save.0, i8** %entry.sp.save.0.ptr, align 4
  %end.call.unwind.frame.cast = bitcast { i8*, { i8*, i8* } }* %end.call.unwind.frame to i8*
  store i8* %end.call.unwind.frame.cast, i8** @"(internal/task).windchain", align 4
  ret i8 undef

end.call.after:                                   ; preds = %end.call, <null operand!>
  store i8* %entry.sp.save.0, i8** @"(internal/task).sp", align 4
  ret i8 %ret
}

define i8 @multiCall(i8* %ptr) {
entry:
  %rewind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %rewind.check = icmp ne i8* %rewind.ptr, null
  br i1 %rewind.check, label %entry.rewind, label %entry.normal

entry.rewind:                                     ; preds = %entry
  %rewind.next.ptr = bitcast i8* %rewind.ptr to i8**
  %rewind.next = load i8*, i8** %rewind.next.ptr, align 4
  store i8* %rewind.next, i8** @"(internal/task).windchain", align 4
  %rewind.pack = bitcast i8* %rewind.ptr to { i8*, i1 }*
  %rewind.pack.idx = getelementptr inbounds { i8*, i1 }, { i8*, i1 }* %rewind.pack, i32 0, i32 1
  %rewind.idx = load i1, i1* %rewind.pack.idx, align 1
  switch i1 %rewind.idx, label %rewind.unreachable [
    i1 false, label %entry.call
    i1 true, label %entry.call.after.call.rewind
  ]

rewind.unreachable:                               ; preds = %entry.rewind
  unreachable

entry.normal:                                     ; preds = %entry
  br label %entry.call

entry.call:                                       ; preds = %entry.rewind, %entry.normal
  %val = call i8 @usePtr(i8* %ptr)
  %entry.call.unwind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %entry.call.unwind.check = icmp ne i8* %entry.call.unwind.ptr, null
  br i1 %entry.call.unwind.check, label %entry.call.unwind, label %entry.call.after

entry.call.unwind:                                ; preds = %entry.call
  %entry.call.unwind.frame.sp = load i8*, i8** @"(internal/task).sp", align 4
  %entry.call.unwind.frame.sp.int = ptrtoint i8* %entry.call.unwind.frame.sp to i32
  %entry.call.unwind.frame.sp.sub = sub i32 %entry.call.unwind.frame.sp.int, 8
  %entry.call.unwind.frame.sp.align = and i32 %entry.call.unwind.frame.sp.sub, -4
  %entry.call.unwind.frame.sp.new = inttoptr i32 %entry.call.unwind.frame.sp.align to i8*
  store i8* %entry.call.unwind.frame.sp.new, i8** @"(internal/task).sp", align 4
  %entry.call.unwind.frame = bitcast i8* %entry.call.unwind.frame.sp.new to { i8*, i1, {} }*
  %entry.call.unwind.frame.prev = getelementptr inbounds { i8*, i1, {} }, { i8*, i1, {} }* %entry.call.unwind.frame, i32 0, i32 0
  store i8* %entry.call.unwind.ptr, i8** %entry.call.unwind.frame.prev, align 4
  %entry.call.unwind.frame.idx = getelementptr inbounds { i8*, i1, {} }, { i8*, i1, {} }* %entry.call.unwind.frame, i32 0, i32 1
  store i1 false, i1* %entry.call.unwind.frame.idx, align 1
  %entry.call.unwind.frame.cast = bitcast { i8*, i1, {} }* %entry.call.unwind.frame to i8*
  store i8* %entry.call.unwind.frame.cast, i8** @"(internal/task).windchain", align 4
  ret i8 undef

entry.call.after:                                 ; preds = %entry.call, <null operand!>
  br label %entry.call.after.call

entry.call.after.call.rewind:                     ; preds = %entry.rewind
  %entry.call.after.call.rewind.superframe = bitcast i8* %rewind.ptr to { i8*, i1, { i8 } }*
  %entry.call.after.call.rewind.frame = getelementptr inbounds { i8*, i1, { i8 } }, { i8*, i1, { i8 } }* %entry.call.after.call.rewind.superframe, i32 0, i32 2
  %val.save.0.reload.ptr = getelementptr inbounds { i8 }, { i8 }* %entry.call.after.call.rewind.frame, i32 0, i32 0
  %val.save.0.reload = load i8, i8* %val.save.0.reload.ptr, align 1
  br label %entry.call.after.call

entry.call.after.call:                            ; preds = %entry.call.after.call.rewind, %entry.call.after
  %val.save.0 = phi i8 [ %val, %entry.call.after ], [ %val.save.0.reload, %entry.call.after.call.rewind ]
  call void @foo()
  %entry.call.after.call.unwind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %entry.call.after.call.unwind.check = icmp ne i8* %entry.call.after.call.unwind.ptr, null
  br i1 %entry.call.after.call.unwind.check, label %entry.call.after.call.unwind, label %entry.call.after.call.after

entry.call.after.call.unwind:                     ; preds = %entry.call.after.call
  %entry.call.after.call.unwind.frame.sp = load i8*, i8** @"(internal/task).sp", align 4
  %entry.call.after.call.unwind.frame.sp.int = ptrtoint i8* %entry.call.after.call.unwind.frame.sp to i32
  %entry.call.after.call.unwind.frame.sp.sub = sub i32 %entry.call.after.call.unwind.frame.sp.int, 8
  %entry.call.after.call.unwind.frame.sp.align = and i32 %entry.call.after.call.unwind.frame.sp.sub, -4
  %entry.call.after.call.unwind.frame.sp.new = inttoptr i32 %entry.call.after.call.unwind.frame.sp.align to i8*
  store i8* %entry.call.after.call.unwind.frame.sp.new, i8** @"(internal/task).sp", align 4
  %entry.call.after.call.unwind.frame = bitcast i8* %entry.call.after.call.unwind.frame.sp.new to { i8*, i1, { i8 } }*
  %entry.call.after.call.unwind.frame.prev = getelementptr inbounds { i8*, i1, { i8 } }, { i8*, i1, { i8 } }* %entry.call.after.call.unwind.frame, i32 0, i32 0
  store i8* %entry.call.after.call.unwind.ptr, i8** %entry.call.after.call.unwind.frame.prev, align 4
  %entry.call.after.call.unwind.frame.idx = getelementptr inbounds { i8*, i1, { i8 } }, { i8*, i1, { i8 } }* %entry.call.after.call.unwind.frame, i32 0, i32 1
  store i1 true, i1* %entry.call.after.call.unwind.frame.idx, align 1
  %entry.call.after.call.unwind.frame.frame = getelementptr inbounds { i8*, i1, { i8 } }, { i8*, i1, { i8 } }* %entry.call.after.call.unwind.frame, i32 0, i32 2
  %val.save.0.ptr = getelementptr inbounds { i8 }, { i8 }* %entry.call.after.call.unwind.frame.frame, i32 0, i32 0
  store i8 %val.save.0, i8* %val.save.0.ptr, align 1
  %entry.call.after.call.unwind.frame.cast = bitcast { i8*, i1, { i8 } }* %entry.call.after.call.unwind.frame to i8*
  store i8* %entry.call.after.call.unwind.frame.cast, i8** @"(internal/task).windchain", align 4
  ret i8 undef

entry.call.after.call.after:                      ; preds = %entry.call.after.call, <null operand!>
  ret i8 %val.save.0
}

define void @savelessLoop(i8* %x) {
entry:
  %rewind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %rewind.check = icmp ne i8* %rewind.ptr, null
  br i1 %rewind.check, label %loop.header.call, label %entry.normal

entry.normal:                                     ; preds = %entry
  br label %loop.header

loop.header:                                      ; preds = %loop.body, %entry.normal
  br label %loop.header.call

loop.header.call:                                 ; preds = %entry, %loop.header
  %next = call i1 @next(i8* %x)
  %unwind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %unwind.check = icmp ne i8* %unwind.ptr, null
  br i1 %unwind.check, label %unwind, label %loop.header.call.after

unwind:                                           ; preds = %loop.header.call
  ret void

loop.header.call.after:                           ; preds = %loop.header.call, <null operand!>
  br i1 %next, label %loop.body, label %end

loop.body:                                        ; preds = %loop.header.call.after
  call void @callNothing()
  br label %loop.header

end:                                              ; preds = %loop.header.call.after
  ret void
}

define void @icall(i8* (i8*, i8*)* %fn.ptr, i8* %fn.ctx, i8* %start) {
entry:
  %rewind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %rewind.check = icmp ne i8* %rewind.ptr, null
  br i1 %rewind.check, label %loop.call.rewind, label %entry.normal

entry.normal:                                     ; preds = %entry
  br label %loop

loop:                                             ; preds = %loop.call.after, %entry.normal
  %ptr = phi i8* [ %start, %entry.normal ], [ %next, %loop.call.after ]
  br label %loop.call

loop.call.rewind:                                 ; preds = %entry
  %rewind.next.ptr = bitcast i8* %rewind.ptr to i8**
  %rewind.next = load i8*, i8** %rewind.next.ptr, align 4
  store i8* %rewind.next, i8** @"(internal/task).windchain", align 4
  %loop.call.rewind.superframe = bitcast i8* %rewind.ptr to { i8*, { i8* } }*
  %loop.call.rewind.frame = getelementptr inbounds { i8*, { i8* } }, { i8*, { i8* } }* %loop.call.rewind.superframe, i32 0, i32 1
  %ptr.save.0.reload.ptr = getelementptr inbounds { i8* }, { i8* }* %loop.call.rewind.frame, i32 0, i32 0
  %ptr.save.0.reload = load i8*, i8** %ptr.save.0.reload.ptr, align 4
  br label %loop.call

loop.call:                                        ; preds = %loop.call.rewind, %loop
  %ptr.save.0 = phi i8* [ %ptr, %loop ], [ %ptr.save.0.reload, %loop.call.rewind ]
  %next = call i8* %fn.ptr(i8* %fn.ctx, i8* %ptr.save.0)
  %loop.call.unwind.ptr = load i8*, i8** @"(internal/task).windchain", align 4
  %loop.call.unwind.check = icmp ne i8* %loop.call.unwind.ptr, null
  br i1 %loop.call.unwind.check, label %loop.call.unwind, label %loop.call.after

loop.call.unwind:                                 ; preds = %loop.call
  %loop.call.unwind.frame.sp = load i8*, i8** @"(internal/task).sp", align 4
  %loop.call.unwind.frame.sp.int = ptrtoint i8* %loop.call.unwind.frame.sp to i32
  %loop.call.unwind.frame.sp.sub = sub i32 %loop.call.unwind.frame.sp.int, 8
  %loop.call.unwind.frame.sp.align = and i32 %loop.call.unwind.frame.sp.sub, -4
  %loop.call.unwind.frame.sp.new = inttoptr i32 %loop.call.unwind.frame.sp.align to i8*
  store i8* %loop.call.unwind.frame.sp.new, i8** @"(internal/task).sp", align 4
  %loop.call.unwind.frame = bitcast i8* %loop.call.unwind.frame.sp.new to { i8*, { i8* } }*
  %loop.call.unwind.frame.prev = getelementptr inbounds { i8*, { i8* } }, { i8*, { i8* } }* %loop.call.unwind.frame, i32 0, i32 0
  store i8* %loop.call.unwind.ptr, i8** %loop.call.unwind.frame.prev, align 4
  %loop.call.unwind.frame.frame = getelementptr inbounds { i8*, { i8* } }, { i8*, { i8* } }* %loop.call.unwind.frame, i32 0, i32 1
  %ptr.save.0.ptr = getelementptr inbounds { i8* }, { i8* }* %loop.call.unwind.frame.frame, i32 0, i32 0
  store i8* %ptr.save.0, i8** %ptr.save.0.ptr, align 4
  %loop.call.unwind.frame.cast = bitcast { i8*, { i8* } }* %loop.call.unwind.frame to i8*
  store i8* %loop.call.unwind.frame.cast, i8** @"(internal/task).windchain", align 4
  ret void

loop.call.after:                                  ; preds = %loop.call, <null operand!>
  %done = icmp eq i8* %next, null
  br i1 %done, label %end, label %loop

end:                                              ; preds = %loop.call.after
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

attributes #0 = { argmemonly nounwind willreturn }
