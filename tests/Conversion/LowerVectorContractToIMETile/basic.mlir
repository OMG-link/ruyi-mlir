// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 vlen=256" -canonicalize -lower-vector-contract-to-ime-tile -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100" -canonicalize -lower-vector-contract-to-ime-tile -split-input-file | FileCheck %s --check-prefix=NODOT
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 vlen=256" -canonicalize -lower-vector-contract-to-ime-tile -flatten-vector-tiles -canonicalize -split-input-file | FileCheck %s --check-prefix=FLAT
//
// On the K1, C(4x4 i32) + ext(A 4x8 i8) x ext(transpose(B 4x8 i8)) is one
// vmadot. An add of the product to a running tile becomes the dot's
// accumulator; after flattening, the dot reads and writes the flat register
// images directly.

#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @accumulate
// CHECK-SAME: (%[[ACC:[a-z0-9]+]]: vector<4x4xi32>, %[[A:[a-z0-9]+]]: vector<4x8xi8>, %[[B:[a-z0-9]+]]: vector<4x8xi8>)
// CHECK-DAG: %[[FA:.+]] = vector.shape_cast %[[A]] : vector<4x8xi8> to vector<32xi8>
// CHECK-DAG: %[[FB:.+]] = vector.shape_cast %[[B]] : vector<4x8xi8> to vector<32xi8>
// CHECK-DAG: %[[FC:.+]] = vector.shape_cast %[[ACC]] : vector<4x4xi32> to vector<16xi32>
// CHECK: ime.tile.vmadot %[[FC]], %[[FA]], %[[FB]]
// CHECK-NOT: vector.contract
// CHECK-NOT: arith.addi
// NODOT: vector.contract
// FLAT-LABEL: func.func @accumulate
// FLAT: ime.tile.vmadot {{.*}} : vector<32xi8>, vector<32xi8> into vector<16xi32>
// FLAT-NOT: vector.transpose
// FLAT-LABEL: func.func @unsigned
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
// CHECK: ime.tile.vmadotu
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
