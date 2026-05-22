// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s | FileCheck %s
//
// Tests for multiple independent contraction chains (Requirements 2.1, 2.3):
//   Two completely independent contraction patterns (no shared ops) in the
//   same function. Both should be identified as tiling roots and tiled
//   independently.
//
// Pattern:
//   Chain 1: bcast_a1 → bcast_b1 → mul1 → reduce1 (output 8x8)
//   Chain 2: bcast_a2 → bcast_b2 → mul2 → reduce2 (output 8x8)
//
// The two chains have separate inputs, separate intermediate ops, and
// separate outputs. No diamond overlap exists.
//
// With tile-m=4 tile-n=4: both chains have M=8 > 4 and N=8 > 4, so both
// should produce scf.for loops after tiling.

#map_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map_bcast_d1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

// CHECK-LABEL: func.func @two_independent_chains
func.func @two_independent_chains(%arg0: tensor<8x1x16xf32>,
                                   %arg1: tensor<1x8x16xf32>,
                                   %arg2: tensor<8x1x16xf32>,
                                   %arg3: tensor<1x8x16xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %cst = arith.constant 0.0 : f32

  // =========================================================================
  // Chain 1: broadcast → multiply → reduce
  // =========================================================================

  // Broadcast %arg0 on dim 1: (8,1,16) → (8,8,16)
  %empty_bcast_a1 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_a1 = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<8x1x16xf32>) outs(%empty_bcast_a1 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Broadcast %arg1 on dim 0: (1,8,16) → (8,8,16)
  %empty_bcast_b1 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_b1 = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x16xf32>) outs(%empty_bcast_b1 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Element-wise multiply (chain 1)
  %empty_mul1 = tensor.empty() : tensor<8x8x16xf32>
  %mul1 = linalg.generic {
    indexing_maps = [#map_identity, #map_identity, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a1, %bcast_b1 : tensor<8x8x16xf32>, tensor<8x8x16xf32>)
    outs(%empty_mul1 : tensor<8x8x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x16xf32>

  // Additive reduce on dim 2 (chain 1 root)
  %empty_out1 = tensor.empty() : tensor<8x8xf32>
  %init1 = linalg.fill ins(%cst : f32) outs(%empty_out1 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduce1 = linalg.reduce ins(%mul1 : tensor<8x8x16xf32>)
                           outs(%init1 : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // =========================================================================
  // Chain 2: broadcast → multiply → reduce (completely independent)
  // =========================================================================

  // Broadcast %arg2 on dim 1: (8,1,16) → (8,8,16)
  %empty_bcast_a2 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_a2 = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg2 : tensor<8x1x16xf32>) outs(%empty_bcast_a2 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Broadcast %arg3 on dim 0: (1,8,16) → (8,8,16)
  %empty_bcast_b2 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_b2 = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg3 : tensor<1x8x16xf32>) outs(%empty_bcast_b2 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Element-wise multiply (chain 2)
  %empty_mul2 = tensor.empty() : tensor<8x8x16xf32>
  %mul2 = linalg.generic {
    indexing_maps = [#map_identity, #map_identity, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a2, %bcast_b2 : tensor<8x8x16xf32>, tensor<8x8x16xf32>)
    outs(%empty_mul2 : tensor<8x8x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x16xf32>

  // Additive reduce on dim 2 (chain 2 root)
  %empty_out2 = tensor.empty() : tensor<8x8xf32>
  %init2 = linalg.fill ins(%cst : f32) outs(%empty_out2 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduce2 = linalg.reduce ins(%mul2 : tensor<8x8x16xf32>)
                           outs(%init2 : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  return %reduce1, %reduce2 : tensor<8x8xf32>, tensor<8x8xf32>
}

// Both chains are independent (no shared ops) → both should be tiled.
// Each chain has output shape 8x8 with tile-m=4 tile-n=4, so both M and N
// dimensions are tiled (8 > 4).
//
// Verify that TWO sets of tiling loops are produced (one per chain).
//
// First chain's tiling loops:
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.reduce
// CHECK:       arith.addf
//
// Second chain's tiling loops:
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.reduce
// CHECK:       arith.addf
