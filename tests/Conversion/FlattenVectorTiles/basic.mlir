// RUN: buddy-opt %s -flatten-vector-tiles -canonicalize -split-input-file | FileCheck %s
//
// n-D vector tiles map onto 1-D registers: a contiguous transfer becomes one
// 1-D transfer of a collapsed memref, a non-contiguous one is split into its
// rows, and elementwise ops and loop-carried values are linearized.

// CHECK-LABEL: func.func @contiguous
// CHECK: %[[S:.+]] = memref.collapse_shape %{{.+}} {{\[}}[0, 1]] : memref<4x8xi8> into memref<32xi8>
// CHECK: %[[X:.+]] = vector.transfer_read %[[S]]{{.*}} : memref<32xi8>, vector<32xi8>
// CHECK: %[[Y:.+]] = arith.andi %[[X]], {{.*}} : vector<32xi8>
// CHECK: vector.transfer_write %[[Y]], {{.*}} : vector<32xi8>, memref<32xi8>
// CHECK-NOT: vector<4x8xi8>
func.func @contiguous(%src: memref<4x8xi8>, %dst: memref<4x8xi8>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i8
  %m = arith.constant dense<15> : vector<4x8xi8>
  %x = vector.transfer_read %src[%c0, %c0], %p {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>
  %y = arith.andi %x, %m : vector<4x8xi8>
  vector.transfer_write %y, %dst[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xi8>, memref<4x8xi8>
  return
}

// -----

// A strided destination is written one row at a time.
// CHECK-LABEL: func.func @strided
// CHECK-COUNT-4: vector.transfer_write {{.*}} : vector<4xi32>, memref<4x4xi32, strided<[16, 1]>>
// CHECK-NOT: vector<4x4xi32>
func.func @strided(%v: vector<4x4xi32>, %dst: memref<4x4xi32, strided<[16, 1]>>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %v, %dst[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xi32>, memref<4x4xi32, strided<[16, 1]>>
  return
}

// -----

// Loop-carried tiles are linearized with the loop.
// CHECK-LABEL: func.func @loop
// CHECK: scf.for {{.*}} -> (vector<16xi32>)
// CHECK: arith.addi {{.*}} : vector<16xi32>
// CHECK-NOT: vector<4x4xi32>
func.func @loop(%x: vector<4x4xi32>, %dst: memref<4x4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %z = arith.constant dense<0> : vector<4x4xi32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %z) -> (vector<4x4xi32>) {
    %s = arith.addi %acc, %x : vector<4x4xi32>
    scf.yield %s : vector<4x4xi32>
  }
  vector.transfer_write %r, %dst[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xi32>, memref<4x4xi32>
  return
}
