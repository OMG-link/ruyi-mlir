// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -convert-tile-tensor-to-vector -canonicalize -split-input-file | FileCheck %s
//
// A Triton masked load arrives as the valid prefix of a view, read through a
// subview and inserted at offset 0 into a fill of the view's shape. It becomes
// one transfer_read of the whole view, padded with the fill value under a mask
// of the subview's sizes.

// The column-0 load of an IME weight-scale operand: a constant mask.
// CHECK-LABEL: func.func @column
// CHECK: %[[V:.+]] = memref.reinterpret_cast
// CHECK: %[[M:.+]] = vector.constant_mask [8, 1] : vector<8x8xi1>
// CHECK: %[[R:.+]] = vector.transfer_read %[[V]][%{{.+}}, %{{.+}}], %{{.+}}, %[[M]] {in_bounds = [true, true]} : memref<8x8xf16, strided<[8, 1]>>, vector<8x8xf16>
// CHECK: vector.transfer_write %[[R]]
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: linalg.fill
func.func @column(%src: memref<f16>, %dst: memref<f16>) {
  %cst = arith.constant 0.0 : f16
  %v = memref.reinterpret_cast %src to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<f16> to memref<8x8xf16, strided<[8, 1]>>
  %sub = memref.subview %v[0, 0] [8, 1] [1, 1] : memref<8x8xf16, strided<[8, 1]>> to memref<8x1xf16, strided<[8, 1]>>
  %t = bufferization.to_tensor %sub restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %e = tensor.empty() : tensor<8x8xf16>
  %f = linalg.fill ins(%cst : f16) outs(%e : tensor<8x8xf16>) -> tensor<8x8xf16>
  %r = tensor.insert_slice %t into %f[0, 0] [8, 1] [1, 1] : tensor<8x1xf16> into tensor<8x8xf16>
  %o = memref.reinterpret_cast %dst to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<f16> to memref<8x8xf16, strided<[8, 1]>>
  bufferization.materialize_in_destination %r in writable %o : (tensor<8x8xf16>, memref<8x8xf16, strided<[8, 1]>>) -> ()
  return
}

// -----

// A run-time bound, `tl.arange(0, 64) < n`, with a non-zero fill.
// CHECK-LABEL: func.func @dynamic
// CHECK: %[[M:.+]] = vector.create_mask %{{.+}} : vector<64xi1>
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}], %{{.+}}, %[[M]] {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
func.func @dynamic(%src: memref<64xf32>, %dst: memref<64xf32>, %n: index) {
  %cst = arith.constant -1.0 : f32
  %sub = memref.subview %src[0] [%n] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
  %t = bufferization.to_tensor %sub restrict : memref<?xf32, strided<[1]>> to tensor<?xf32>
  %e = tensor.empty() : tensor<64xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<64xf32>) -> tensor<64xf32>
  %r = tensor.insert_slice %t into %f[0] [%n] [1] : tensor<?xf32> into tensor<64xf32>
  bufferization.materialize_in_destination %r in writable %dst : (tensor<64xf32>, memref<64xf32>) -> ()
  return
}

// -----

// A slice inserted away from the origin is not a prefix mask: it stays.
// CHECK-LABEL: func.func @offset
// CHECK: tensor.insert_slice
// CHECK-NOT: vector.transfer_read
func.func @offset(%src: memref<8x8xf16>, %dst: memref<8x8xf16>) {
  %cst = arith.constant 0.0 : f16
  %sub = memref.subview %src[0, 0] [8, 1] [1, 1] : memref<8x8xf16> to memref<8x1xf16, strided<[8, 1]>>
  %t = bufferization.to_tensor %sub restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %e = tensor.empty() : tensor<8x8xf16>
  %f = linalg.fill ins(%cst : f16) outs(%e : tensor<8x8xf16>) -> tensor<8x8xf16>
  %r = tensor.insert_slice %t into %f[0, 3] [8, 1] [1, 1] : tensor<8x1xf16> into tensor<8x8xf16>
  bufferization.materialize_in_destination %r in writable %dst : (tensor<8x8xf16>, memref<8x8xf16>) -> ()
  return
}

// -----

// Fills are shared: CSE merges those of several masked loads, and one may
// also be read by something else. Each masked load converts on its own; the
// fill stays for its other reader.
// CHECK-LABEL: func.func @shared_fill
// CHECK-COUNT-2: vector.transfer_read {{.+}}, %{{.+}}, %{{.+}} {in_bounds = [true, true]} : memref<8x8xf16>, vector<8x8xf16>
// CHECK-NOT: tensor.insert_slice
func.func @shared_fill(%src: memref<8x8xf16>, %src2: memref<8x8xf16>, %dst: memref<8x8xf16>,
                       %dst2: memref<8x8xf16>, %dst3: memref<8x8xf16>) {
  %cst = arith.constant 0.0 : f16
  %sub = memref.subview %src[0, 0] [8, 1] [1, 1] : memref<8x8xf16> to memref<8x1xf16, strided<[8, 1]>>
  %t = bufferization.to_tensor %sub restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %sub2 = memref.subview %src2[0, 0] [8, 1] [1, 1] : memref<8x8xf16> to memref<8x1xf16, strided<[8, 1]>>
  %t2 = bufferization.to_tensor %sub2 restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %e = tensor.empty() : tensor<8x8xf16>
  %f = linalg.fill ins(%cst : f16) outs(%e : tensor<8x8xf16>) -> tensor<8x8xf16>
  %r = tensor.insert_slice %t into %f[0, 0] [8, 1] [1, 1] : tensor<8x1xf16> into tensor<8x8xf16>
  %r2 = tensor.insert_slice %t2 into %f[0, 0] [8, 1] [1, 1] : tensor<8x1xf16> into tensor<8x8xf16>
  bufferization.materialize_in_destination %r in writable %dst : (tensor<8x8xf16>, memref<8x8xf16>) -> ()
  bufferization.materialize_in_destination %r2 in writable %dst2 : (tensor<8x8xf16>, memref<8x8xf16>) -> ()
  bufferization.materialize_in_destination %f in writable %dst3 : (tensor<8x8xf16>, memref<8x8xf16>) -> ()
  return
}
