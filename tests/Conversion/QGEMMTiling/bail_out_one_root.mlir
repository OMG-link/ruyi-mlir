// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s 2>&1 | FileCheck %s
//
// Tests for bail-out behavior with one valid and one invalid root
// (Requirement 12.1, 12.2, 12.5):
//   Two independent contraction patterns exist in the function.
//   One has dynamic M dimension (invalid for tiling) — it should be skipped.
//   The other has static dimensions larger than tile sizes — it should be tiled.
//   The pass should succeed without error.

#map_identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d0  = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map_bcast_d1  = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

// The dynamic root should emit a remark about being skipped.
// CHECK: remark{{.*}}dynamic M/N dimension in tiling root; skipping

// CHECK-LABEL: func.func @bail_out_one_root
func.func @bail_out_one_root(%arg0: tensor<8x1x16xf32>,
                              %arg1: tensor<1x8x16xf32>,
                              %dyn_arg0: tensor<?x1x16xf32>,
                              %dyn_arg1: tensor<1x8x16xf32>) -> (tensor<8x8xf32>, tensor<?x8xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %dyn_arg0, %c0 : tensor<?x1x16xf32>

  // =========================================================================
  // Chain 1: VALID — static 8x8 output, tile-m=4 tile-n=4 → both tiled
  // =========================================================================

  // Broadcast %arg0 on dim 1: (8,1,16) → (8,8,16)
  %empty0 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<8x1x16xf32>) outs(%empty0 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Broadcast %arg1 on dim 0: (1,8,16) → (8,8,16)
  %empty1 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x16xf32>) outs(%empty1 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Element-wise multiply
  %empty2 = tensor.empty() : tensor<8x8x16xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity3, #map_identity3, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<8x8x16xf32>, tensor<8x8x16xf32>)
    outs(%empty2 : tensor<8x8x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x16xf32>

  // Reduce on dim 2 — tiling root for chain 1
  %empty3 = tensor.empty() : tensor<8x8xf32>
  %init1 = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduce1 = linalg.reduce ins(%mul : tensor<8x8x16xf32>)
                           outs(%init1 : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // =========================================================================
  // Chain 2: INVALID — dynamic M dimension → should be skipped
  // =========================================================================

  // Broadcast %dyn_arg0 on dim 1: (?,1,16) → (?,8,16)
  %empty4 = tensor.empty(%dim) : tensor<?x8x16xf32>
  %dyn_bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%dyn_arg0 : tensor<?x1x16xf32>) outs(%empty4 : tensor<?x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<?x8x16xf32>

  // Broadcast %dyn_arg1 on dim 0: (1,8,16) → (?,8,16)
  %empty5 = tensor.empty(%dim) : tensor<?x8x16xf32>
  %dyn_bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%dyn_arg1 : tensor<1x8x16xf32>) outs(%empty5 : tensor<?x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<?x8x16xf32>

  // Element-wise multiply (dynamic)
  %empty6 = tensor.empty(%dim) : tensor<?x8x16xf32>
  %dyn_mul = linalg.generic {
    indexing_maps = [#map_identity3, #map_identity3, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%dyn_bcast_a, %dyn_bcast_b : tensor<?x8x16xf32>, tensor<?x8x16xf32>)
    outs(%empty6 : tensor<?x8x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<?x8x16xf32>

  // Reduce on dim 2 — tiling root for chain 2 (dynamic M → skipped)
  %empty7 = tensor.empty(%dim) : tensor<?x8xf32>
  %init2 = linalg.fill ins(%cst : f32) outs(%empty7 : tensor<?x8xf32>) -> tensor<?x8xf32>
  %reduce2 = linalg.reduce ins(%dyn_mul : tensor<?x8x16xf32>)
                           outs(%init2 : tensor<?x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  return %reduce1, %reduce2 : tensor<8x8xf32>, tensor<?x8xf32>
}

// The valid chain (8x8 output, tile-m=4 tile-n=4) should be tiled.
// CHECK: scf.for
// CHECK:   scf.for

// The dynamic chain's reduce should still be present (untiled, outside loops).
// CHECK: linalg.reduce
// CHECK:   arith.addf
