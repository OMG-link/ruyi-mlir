// RUN: buddy-opt --qgemm-decompress-coupling %s 2>&1 | FileCheck %s
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s
//
// Req 14.7, 14.8, 14.11, 14.12: Load_Copy_Forwarding is a context-free
// pattern applied over the whole function. None of these functions carries a
// QGEMM annotation, so every load copy lies outside every Scope. A copy is
// forwarded only when nothing in the function can modify the copied memory;
// otherwise it is kept, and no remark is emitted either way. How the
// forwarded tensor is used does not matter: it is never `writable`, so
// bufferization never writes the source through it.
//
// CHECK-NOT: remark

#id = affine_map<(d0, d1) -> (d0, d1)>

// (1) Read-only source: the alloc and copy are erased, the to_tensor reads
// the view and loses `writable`; its discardable attributes are kept.
// CHECK-LABEL: func.func @forwarded
// CHECK: %[[V:[a-z0-9_]+]] = memref.reinterpret_cast
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: bufferization.to_tensor %[[V]] restrict {test.tag = 1 : i64} :
// CHECK-NOT: writable
// CHECK: return
func.func @forwarded(%arg0: memref<*xi8>) -> tensor<8x64xi32> {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable {test.tag = 1 : i64} : memref<8x64xi8> to tensor<8x64xi8>
  %e = tensor.empty() : tensor<8x64xi32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<8x64xi8>) outs(%e : tensor<8x64xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x64xi32>
  return %r : tensor<8x64xi32>
}

// (2) Another view of the same base is written after the use: kept.
// CHECK-LABEL: func.func @written_alias
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable
func.func @written_alias(%arg0: memref<*xi8>) -> tensor<8x64xi32> {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %e = tensor.empty() : tensor<8x64xi32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<8x64xi8>) outs(%e : tensor<8x64xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x64xi32>
  %wv = memref.reinterpret_cast %arg0 to offset: [512], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1], offset: 512>>
  %c = arith.constant 0 : i8
  %i0 = arith.constant 0 : index
  memref.store %c, %wv[%i0, %i0] : memref<8x64xi8, strided<[64, 1], offset: 512>>
  return %r : tensor<8x64xi32>
}

// (3) Another memref argument is written and may alias %arg0: kept.
// CHECK-LABEL: func.func @other_arg_written
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable
func.func @other_arg_written(%arg0: memref<*xi8>, %arg1: memref<*xi32>) {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %e = tensor.empty() : tensor<8x64xi32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<8x64xi8>) outs(%e : tensor<8x64xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x64xi32>
  %out = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi32> to memref<8x64xi32, strided<[64, 1]>>
  bufferization.materialize_in_destination %r in writable %out : (tensor<8x64xi32>, memref<8x64xi32, strided<[64, 1]>>) -> ()
  return
}

// (3b) The same write, but the written argument is `llvm.noalias`: it cannot
// alias %arg0, so the copy is forwarded.
// CHECK-LABEL: func.func @other_arg_written_noalias
// CHECK-NOT: memref.alloc
// CHECK: bufferization.to_tensor %{{[a-z0-9_]+}} restrict :
// CHECK: bufferization.materialize_in_destination
func.func @other_arg_written_noalias(%arg0: memref<*xi8>, %arg1: memref<*xi32> {llvm.noalias}) {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %e = tensor.empty() : tensor<8x64xi32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<8x64xi8>) outs(%e : tensor<8x64xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x64xi32>
  %out = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi32> to memref<8x64xi32, strided<[64, 1]>>
  bufferization.materialize_in_destination %r in writable %out : (tensor<8x64xi32>, memref<8x64xi32, strided<[64, 1]>>) -> ()
  return
}

// (4) The tensor is a destination (outs) operand: forwarded all the same.
// The non-writable to_tensor makes bufferization allocate a new buffer for
// the in-place write (see forward_copy_bufferize.mlir).
// CHECK-LABEL: func.func @destination_use_forwarded
// CHECK: %[[V:[a-z0-9_]+]] = memref.reinterpret_cast
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: bufferization.to_tensor %[[V]] restrict :
// CHECK: outs(%{{[a-z0-9_]+}} : tensor<8x64xi8>)
func.func @destination_use_forwarded(%arg0: memref<*xi8>, %src: tensor<8x64xi8>) -> tensor<8x64xi8> {
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

// (5) A call receives the base; its effects are unknown: kept.
// CHECK-LABEL: func.func @call_use
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable
func.func private @opaque(memref<*xi8>)
func.func @call_use(%arg0: memref<*xi8>) -> tensor<8x64xi32> {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %e = tensor.empty() : tensor<8x64xi32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<8x64xi8>) outs(%e : tensor<8x64xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x64xi32>
  func.call @opaque(%arg0) : (memref<*xi8>) -> ()
  return %r : tensor<8x64xi32>
}
