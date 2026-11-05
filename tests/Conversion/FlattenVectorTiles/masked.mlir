// RUN: buddy-opt %s -flatten-vector-tiles | FileCheck %s
//
// A masked contiguous transfer flattens to a 1-D masked transfer that reads
// the n-D mask through a shape_cast; the n-D create_mask and constant_mask
// stay as they are rather than failing the linearization.

// CHECK-LABEL: func.func @masked_read
// CHECK: %[[MASK:.+]] = vector.create_mask %{{.+}}, %{{.+}} : vector<8x8xi1>
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[MASK]] : vector<8x8xi1> to vector<64xi1>
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}], %{{.+}}, %[[FLAT]] {in_bounds = [true]} : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
func.func @masked_read(%m: memref<8x8xf16, strided<[8, 1], offset: ?>>, %r: index, %k: index) -> vector<8x8xf16> {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f16
  %mask = vector.create_mask %r, %k : vector<8x8xi1>
  %v = vector.transfer_read %m[%c0, %c0], %p, %mask {in_bounds = [true, true]} : memref<8x8xf16, strided<[8, 1], offset: ?>>, vector<8x8xf16>
  return %v : vector<8x8xf16>
}

// CHECK-LABEL: func.func @const_masked_write
// CHECK: %[[MASK:.+]] = vector.constant_mask [8, 1] : vector<8x8xi1>
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[MASK]] : vector<8x8xi1> to vector<64xi1>
// CHECK: vector.transfer_write %{{.+}}, %{{.+}}[%{{.+}}], %[[FLAT]] {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
func.func @const_masked_write(%m: memref<8x8xf32>, %v: vector<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [8, 1] : vector<8x8xi1>
  vector.transfer_write %v, %m[%c0, %c0], %mask {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
  return
}
