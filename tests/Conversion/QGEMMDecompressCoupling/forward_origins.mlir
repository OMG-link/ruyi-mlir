// RUN: buddy-opt --qgemm-decompress-coupling %s 2>&1 | FileCheck %s
//
// Req 12.28, 14.13: Load_Copy_Forwarding accepts any memref copy source whose
// Memref_Origin is supported (entry-block argument, local alloc/alloca,
// constant global) and keeps the copy for any other origin. A local origin is
// modified by any write or free tracing to the same alloc. No remark is
// emitted either way.
//
// CHECK-NOT: remark

memref.global "private" constant @cst : memref<8x64xi8> = dense<1>
memref.global "private" @var : memref<8x64xi8> = dense<1>

// An unwritten local alloc source: forwarded, only the source alloc stays.
// CHECK-LABEL: func.func @local_unwritten
// CHECK: %[[SRC:[a-z0-9_]+]] = memref.alloc
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: bufferization.to_tensor %[[SRC]] restrict :
func.func @local_unwritten() -> tensor<8x64xi8> {
  %src = memref.alloc() : memref<8x64xi8>
  %a = memref.alloc() : memref<8x64xi8>
  memref.copy %src, %a : memref<8x64xi8> to memref<8x64xi8>
  %t = bufferization.to_tensor %a restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  return %t : tensor<8x64xi8>
}

// A local alloc source written through a view of it: kept.
// CHECK-LABEL: func.func @local_written
// CHECK: memref.alloc
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable
func.func @local_written(%x: i8) -> tensor<8x64xi8> {
  %c0 = arith.constant 0 : index
  %src = memref.alloc() : memref<8x64xi8>
  %flat = memref.collapse_shape %src [[0, 1]] : memref<8x64xi8> into memref<512xi8>
  memref.store %x, %flat[%c0] : memref<512xi8>
  %a = memref.alloc() : memref<8x64xi8>
  memref.copy %src, %a : memref<8x64xi8> to memref<8x64xi8>
  %t = bufferization.to_tensor %a restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  return %t : tensor<8x64xi8>
}

// A constant global: forwarded.
// CHECK-LABEL: func.func @constant_global
// CHECK: %[[G:[a-z0-9_]+]] = memref.get_global @cst
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: bufferization.to_tensor %[[G]] restrict :
func.func @constant_global() -> tensor<8x64xi8> {
  %g = memref.get_global @cst : memref<8x64xi8>
  %a = memref.alloc() : memref<8x64xi8>
  memref.copy %g, %a : memref<8x64xi8> to memref<8x64xi8>
  %t = bufferization.to_tensor %a restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  return %t : tensor<8x64xi8>
}

// A non-constant global: kept.
// CHECK-LABEL: func.func @mutable_global
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable
func.func @mutable_global() -> tensor<8x64xi8> {
  %g = memref.get_global @var : memref<8x64xi8>
  %a = memref.alloc() : memref<8x64xi8>
  memref.copy %g, %a : memref<8x64xi8> to memref<8x64xi8>
  %t = bufferization.to_tensor %a restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  return %t : tensor<8x64xi8>
}

// A memref.subview + memref.cast source chain over an argument: forwarded.
// CHECK-LABEL: func.func @subview_cast_chain
// CHECK: %[[SV:[a-z0-9_]+]] = memref.subview
// CHECK: %[[C:[a-z0-9_]+]] = memref.cast %[[SV]]
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: bufferization.to_tensor %[[C]] restrict :
func.func @subview_cast_chain(%arg0: memref<16x64xi8> {llvm.noalias}) -> tensor<8x64xi8> {
  %sv = memref.subview %arg0[0, 0] [8, 64] [1, 1] : memref<16x64xi8> to memref<8x64xi8, strided<[64, 1]>>
  %c = memref.cast %sv : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8, strided<[?, ?], offset: ?>>
  %a = memref.alloc() : memref<8x64xi8>
  memref.copy %c, %a : memref<8x64xi8, strided<[?, ?], offset: ?>> to memref<8x64xi8>
  %t = bufferization.to_tensor %a restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  return %t : tensor<8x64xi8>
}
