// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector='vector-width=4' | FileCheck %s
//
// LinalgToVIR aligns a rank-extending broadcast input with the iteration space
// by creating a memref.expand_shape with a fully static output shape inside
// vir.set_vl. VIRToVector clones that op into the vector loop. The clone must
// keep the static output shape: getOutputShape() returns only the dynamic
// operands, so building the clone from it produced an expand_shape with no
// shape bounds, which the verifier rejects.

#in = affine_map<(d0, d1) -> (d1)>
#out = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @rank_extending_broadcast
// CHECK: memref.expand_shape %{{[a-z0-9_]+}} {{\[\[}}0, 1]] output_shape [1, 8] : memref<8xf32> into memref<1x8xf32>
func.func @rank_extending_broadcast(%src: memref<8xf32>, %dst: memref<4x8xf32>) {
  linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "parallel"]} ins(%src : memref<8xf32>) outs(%dst : memref<4x8xf32>) {
  ^bb0(%a: f32, %b: f32):
    linalg.yield %a : f32
  }
  return
}
