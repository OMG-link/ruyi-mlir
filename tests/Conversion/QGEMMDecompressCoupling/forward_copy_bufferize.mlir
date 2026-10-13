// RUN: buddy-opt --qgemm-decompress-coupling %s | buddy-opt --empty-tensor-to-alloc-tensor --one-shot-bufferize=allow-return-allocs-from-loops=true | FileCheck %s
//
// Req 14.7: Load_Copy_Forwarding puts no restriction on how the forwarded
// tensor is used. The replacement to_tensor is never `writable`, so One-Shot
// Bufferize (with the options compiler.py uses) treats the argument's view as
// read-only and gives every in-place write a fresh memref.alloc, copying the
// view into it when the old value is read. None of these functions carries a
// QGEMM annotation, so every load copy lies outside every Scope.
//
// After bufferization no write -- a generic outs, a memref.copy target or a
// memref.store -- targets a memref derived from %arg0 (the `%[[V]]` view).
#id = affine_map<(d0, d1) -> (d0, d1)>

// (a) outs init whose body ignores %out: the generic writes a fresh alloc;
// the view is not even copied, since its value is never read.
// CHECK-LABEL: func.func @outs_ignored
// CHECK: %[[V:[a-z0-9_]+]] = memref.reinterpret_cast %arg0
// CHECK: %[[A:[a-z0-9_]+]] = memref.alloc
// CHECK-NOT: memref.copy %{{[a-z0-9_]+}}, %[[V]]
// CHECK-NOT: outs(%[[V]]
// CHECK: linalg.generic {{.*}} outs(%[[A]] :
// CHECK-NOT: outs(%[[V]]
// CHECK-NOT: memref.store {{.*}}, %[[V]]
// CHECK: return
func.func @outs_ignored(%arg0: memref<*xi8>, %src: tensor<8x64xi8>) -> tensor<8x64xi8> {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<8x64xi8>) outs(%t : tensor<8x64xi8>) {
  ^bb0(%i: i8, %o: i8):
    linalg.yield %i : i8
  } -> tensor<8x64xi8>
  return %r : tensor<8x64xi8>
}

// (b) outs init whose body reads %out: the view is copied into a fresh alloc,
// which the generic then updates.
// CHECK-LABEL: func.func @outs_read
// CHECK: %[[V:[a-z0-9_]+]] = memref.reinterpret_cast %arg0
// CHECK: %[[A:[a-z0-9_]+]] = memref.alloc
// CHECK: memref.copy %[[V]], %[[A]]
// CHECK-NOT: outs(%[[V]]
// CHECK: linalg.generic {{.*}} outs(%[[A]] :
// CHECK-NOT: outs(%[[V]]
// CHECK-NOT: memref.copy %{{[a-z0-9_]+}}, %[[V]]
// CHECK: return
func.func @outs_read(%arg0: memref<*xi8>, %src: tensor<8x64xi8>) -> tensor<8x64xi8> {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<8x64xi8>) outs(%t : tensor<8x64xi8>) {
  ^bb0(%i: i8, %o: i8):
    %x = arith.addi %i, %o : i8
    linalg.yield %x : i8
  } -> tensor<8x64xi8>
  return %r : tensor<8x64xi8>
}

// (c) scf.for iter_arg written in the loop: the loop carries a fresh alloc
// initialised from the view; the in-loop write targets that alloc.
// CHECK-LABEL: func.func @iter_arg
// CHECK: %[[V:[a-z0-9_]+]] = memref.reinterpret_cast %arg0
// CHECK: %[[A:[a-z0-9_]+]] = memref.alloc
// CHECK: memref.copy %[[V]], %[[A]]
// CHECK: scf.for {{.*}} iter_args(%{{[a-z0-9_]+}} = %[[A]])
// CHECK-NOT: outs(%[[V]]
// CHECK-NOT: memref.copy %{{[a-z0-9_]+}}, %[[V]]
// CHECK-NOT: memref.store {{.*}}, %[[V]]
// CHECK: return
func.func @iter_arg(%arg0: memref<*xi8>, %src: tensor<8x64xi8>) -> tensor<8x64xi8> {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %t) -> (tensor<8x64xi8>) {
    %n = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%src : tensor<8x64xi8>) outs(%acc : tensor<8x64xi8>) {
    ^bb0(%i: i8, %o: i8):
      %x = arith.addi %i, %o : i8
      linalg.yield %x : i8
    } -> tensor<8x64xi8>
    scf.yield %n : tensor<8x64xi8>
  }
  return %res : tensor<8x64xi8>
}

// (d) Returned from the function. Without function-boundary bufferization
// (compiler.py does not enable it) the return keeps a tensor operand that
// reads the view; nothing writes the view.
// CHECK-LABEL: func.func @returned
// CHECK: %[[V:[a-z0-9_]+]] = memref.reinterpret_cast %arg0
// CHECK-NOT: outs(%[[V]]
// CHECK-NOT: memref.copy %{{[a-z0-9_]+}}, %[[V]]
// CHECK-NOT: memref.store {{.*}}, %[[V]]
// CHECK: return
func.func @returned(%arg0: memref<*xi8>) -> tensor<8x64xi8> {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  return %t : tensor<8x64xi8>
}
