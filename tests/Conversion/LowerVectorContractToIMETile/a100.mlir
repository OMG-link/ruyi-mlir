// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -canonicalize -lower-vector-contract-to-ime-tile -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 vlen=256" -canonicalize -lower-vector-contract-to-ime-tile -split-input-file | FileCheck %s --check-prefix=K1
//
// On the A100 (XSMTVDotII, VLEN 1024) an int8 8x16 x (8x16)^T product into
// 8x8 int32 is one vmadot, and an fp16 8x8 x (8x8)^T product extended into
// 8x8 fp32 is one vfwmadot. The K1 has neither shape.

#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @int8
// CHECK: ime.tile.vmadot {{.*}} : vector<128xi8>, vector<128xi8> into vector<64xi32>
// CHECK-NOT: vector.contract
// K1-LABEL: func.func @int8
// K1: vector.contract
func.func @int8(%acc: vector<8x8xi32>, %a: vector<8x16xi8>, %b: vector<8x16xi8>) -> vector<8x8xi32> {
  %bt = vector.transpose %b, [1, 0] : vector<8x16xi8> to vector<16x8xi8>
  %ea = arith.extsi %a : vector<8x16xi8> to vector<8x16xi32>
  %eb = arith.extsi %bt : vector<16x8xi8> to vector<16x8xi32>
  %d = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<8x16xi32>, vector<16x8xi32> into vector<8x8xi32>
  return %d : vector<8x8xi32>
}

// -----

#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @fp16
// CHECK: ime.tile.vfwmadot {{.*}} : vector<64xf16>, vector<64xf16> into vector<64xf32>
// CHECK-NOT: vector.contract
// K1-LABEL: func.func @fp16
// K1: vector.contract
func.func @fp16(%acc: vector<8x8xf32>, %a: vector<8x8xf16>, %b: vector<8x8xf16>) -> vector<8x8xf32> {
  %bt = vector.transpose %b, [1, 0] : vector<8x8xf16> to vector<8x8xf16>
  %ea = arith.extf %a : vector<8x8xf16> to vector<8x8xf32>
  %eb = arith.extf %bt : vector<8x8xf16> to vector<8x8xf32>
  %d = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %d : vector<8x8xf32>
}
