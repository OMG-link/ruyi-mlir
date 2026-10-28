// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60" -canonicalize -lower-vector-contract-to-ime-tile -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100" -canonicalize -lower-vector-contract-to-ime-tile -split-input-file | FileCheck %s --check-prefix=NODOT
//
// A named K1 supplies only a minimum VLEN. Do not specialize a fixed tile
// without an exact VLEN, even when the minimum happens to equal 256.

#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>

// NODOT: vector.contract
// CHECK-LABEL: func.func @accumulate
// CHECK: vector.contract
// CHECK-NOT: ime.tile
func.func @accumulate(%acc: vector<4x4xi32>, %a: vector<4x8xi8>, %b: vector<4x8xi8>) -> vector<4x4xi32> {
  %z = arith.constant dense<0> : vector<4x4xi32>
  %bt = vector.transpose %b, [1, 0] : vector<4x8xi8> to vector<8x4xi8>
  %ea = arith.extsi %a : vector<4x8xi8> to vector<4x8xi32>
  %eb = arith.extsi %bt : vector<8x4xi8> to vector<8x4xi32>
  %d = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %z : vector<4x8xi32>, vector<8x4xi32> into vector<4x4xi32>
  %r = arith.addi %acc, %d : vector<4x4xi32>
  return %r : vector<4x4xi32>
}

// -----

#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @unsigned
// CHECK: vector.contract
// CHECK-NOT: ime.tile
func.func @unsigned(%acc: vector<4x4xi32>, %a: vector<4x8xi8>, %b: vector<4x8xi8>) -> vector<4x4xi32> {
  %bt = vector.transpose %b, [1, 0] : vector<4x8xi8> to vector<8x4xi8>
  %ea = arith.extui %a : vector<4x8xi8> to vector<4x8xi32>
  %eb = arith.extui %bt : vector<8x4xi8> to vector<8x4xi32>
  %d = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<4x8xi32>, vector<8x4xi32> into vector<4x4xi32>
  return %d : vector<4x4xi32>
}

// -----

#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>

// Mixed signedness has no single instruction here.
// CHECK-LABEL: func.func @mixed
// CHECK: vector.contract
// CHECK-NOT: ime.tile
func.func @mixed(%acc: vector<4x4xi32>, %a: vector<4x8xi8>, %b: vector<4x8xi8>) -> vector<4x4xi32> {
  %bt = vector.transpose %b, [1, 0] : vector<4x8xi8> to vector<8x4xi8>
  %ea = arith.extsi %a : vector<4x8xi8> to vector<4x8xi32>
  %eb = arith.extui %bt : vector<8x4xi8> to vector<8x4xi32>
  %d = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<4x8xi32>, vector<8x4xi32> into vector<4x4xi32>
  return %d : vector<4x4xi32>
}
