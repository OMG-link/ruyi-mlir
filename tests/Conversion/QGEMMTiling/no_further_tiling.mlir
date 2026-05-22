// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s | FileCheck %s
//
// Tests for no-further-tiling behavior (Requirements 12.2, 12.5):
//   The contraction pattern has M=4 and N=4 output dimensions, which are
//   equal to the requested tile sizes (tile-m=4, tile-n=4). Since both
//   extents ≤ tile sizes, no tiling should occur. The pass should return
//   success and leave the IR unchanged.
//
// This simulates the scenario where a previous tiling pass has already
// reduced the extents to the tile size, and a subsequent invocation with
// the same tile sizes should be a no-op.

#map_identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d0  = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map_bcast_d1  = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

// CHECK-LABEL: func.func @no_further_tiling
func.func @no_further_tiling(%arg0: tensor<4x1x8xf32>,
                              %arg1: tensor<1x4x8xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant 0.0 : f32

  // Broadcast %arg0 on dim 1: (4,1,8) → (4,4,8)
  %empty0 = tensor.empty() : tensor<4x4x8xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<4x1x8xf32>) outs(%empty0 : tensor<4x4x8xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x8xf32>

  // Broadcast %arg1 on dim 0: (1,4,8) → (4,4,8)
  %empty1 = tensor.empty() : tensor<4x4x8xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x4x8xf32>) outs(%empty1 : tensor<4x4x8xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x8xf32>

  // Element-wise multiply
  %empty2 = tensor.empty() : tensor<4x4x8xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity3, #map_identity3, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<4x4x8xf32>, tensor<4x4x8xf32>)
    outs(%empty2 : tensor<4x4x8xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<4x4x8xf32>

  // Reduce on dim 2 — this is the tiling root
  // Output shape is 4x4. With tile-m=4, tile-n=4:
  //   M=4 ≤ 4 → tile size = 0
  //   N=4 ≤ 4 → tile size = 0
  //   All zero → skip (no tiling)
  %empty3 = tensor.empty() : tensor<4x4xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %reduced = linalg.reduce ins(%mul : tensor<4x4x8xf32>)
                           outs(%init : tensor<4x4xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  return %reduced : tensor<4x4xf32>
}

// Verify no tiling loop is generated (all dims ≤ tile sizes → IR unchanged).
// CHECK-NOT: scf.for

// Verify original ops are preserved.
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.reduce
// CHECK:   arith.addf
