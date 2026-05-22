// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s | FileCheck %s
//
// NOTE: This test requires the v02 pass restructuring (task 6.1) which replaces
// FinalConsumerFinder with ReduceChainAnalysis. Until that restructuring is
// complete, this test will not pass.
//
// Tests for reduce chain root identification (Requirements 1.2, 1.4, 1.6):
//   1. Two-reduce chain: outermost reduce is selected as root and tiled
//   2. Single reduce: identified as root and tiled
//   3. Non-additive reduce (arith.maxnumf): NOT identified as root, IR unchanged

// ============================================================================
// Test 1: Two-reduce chain — outermost reduce is the root
// ============================================================================
// Pattern: broadcast → multiply → reduce(dim=3, addf) → scale → reduce(dim=2, addf)
// The inner reduce (dim=3) feeds the outer reduce (dim=2) through an
// element-wise scale generic. The outer reduce (dim=2) should be identified
// as the chain root and tiled. Output shape is 8x8 with tile-m=4 tile-n=4,
// so both M=8 and N=8 are > 4 and get tiled.

#map_identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_bcast_d0  = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map_bcast_d1  = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map_identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @two_reduce_chain
func.func @two_reduce_chain(%arg0: tensor<8x1x2x16xf32>,
                             %arg1: tensor<1x8x2x16xf32>) -> tensor<8x8xf32> {
  // Broadcast %arg0 on dim 1
  %empty0 = tensor.empty() : tensor<8x8x2x16xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<8x1x2x16xf32>) outs(%empty0 : tensor<8x8x2x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x2x16xf32>

  // Broadcast %arg1 on dim 0
  %empty1 = tensor.empty() : tensor<8x8x2x16xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x2x16xf32>) outs(%empty1 : tensor<8x8x2x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x2x16xf32>

  // Element-wise multiply (4D)
  %empty2 = tensor.empty() : tensor<8x8x2x16xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity4, #map_identity4, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<8x8x2x16xf32>, tensor<8x8x2x16xf32>)
    outs(%empty2 : tensor<8x8x2x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x2x16xf32>

  // Inner reduce on dim 3 (QK dimension, additive combiner)
  %cst = arith.constant 0.0 : f32
  %empty3 = tensor.empty() : tensor<8x8x2xf32>
  %init3 = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<8x8x2xf32>) -> tensor<8x8x2xf32>
  %inner_reduce = linalg.reduce ins(%mul : tensor<8x8x2x16xf32>)
                                outs(%init3 : tensor<8x8x2xf32>)
                                dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Element-wise scale (connects inner reduce to outer reduce)
  %empty4 = tensor.empty() : tensor<8x8x2xf32>
  %scale = linalg.generic {
    indexing_maps = [#map_identity3, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner_reduce : tensor<8x8x2xf32>) outs(%empty4 : tensor<8x8x2xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    %two = arith.constant 2.0 : f32
    %scaled = arith.mulf %in, %two : f32
    linalg.yield %scaled : f32
  } -> tensor<8x8x2xf32>

  // Outer reduce on dim 2 (KB dimension, additive combiner) — chain root
  %empty5 = tensor.empty() : tensor<8x8xf32>
  %init5 = linalg.fill ins(%cst : f32) outs(%empty5 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %outer_reduce = linalg.reduce ins(%scale : tensor<8x8x2xf32>)
                                outs(%init5 : tensor<8x8xf32>)
                                dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  return %outer_reduce : tensor<8x8xf32>
}

// The outermost reduce (dim=2, output tensor<8x8xf32>) is the chain root.
// With tile-m=4 tile-n=4: M=8 > 4 → tiled, N=8 > 4 → tiled.
// Verify scf.for appears (tiling happened on the root).
// CHECK: scf.for
// Verify fused producers inside the loop body.
// CHECK:   linalg.generic
// CHECK:   linalg.reduce
// CHECK:     arith.addf
// CHECK:   linalg.reduce
// CHECK:     arith.addf


// ============================================================================
// Test 2: Single reduce — identified as root and tiled
// ============================================================================
// A standalone additive reduce with output shape 8x8 should be identified
// as root and tiled with tile-m=4 tile-n=4.

#map_identity3b = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d1_3 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map_bcast_d0_3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>

// CHECK-LABEL: func.func @single_reduce_root
func.func @single_reduce_root(%arg0: tensor<8x1x16xf32>,
                               %arg1: tensor<1x8x16xf32>) -> tensor<8x8xf32> {
  // Broadcast %arg0 on dim 1
  %empty0 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1_3, #map_identity3b],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<8x1x16xf32>) outs(%empty0 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Broadcast %arg1 on dim 0
  %empty1 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0_3, #map_identity3b],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x16xf32>) outs(%empty1 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Element-wise multiply
  %empty2 = tensor.empty() : tensor<8x8x16xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity3b, #map_identity3b, #map_identity3b],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<8x8x16xf32>, tensor<8x8x16xf32>)
    outs(%empty2 : tensor<8x8x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x16xf32>

  // Single additive reduce on dim 2 — this is the root
  %cst = arith.constant 0.0 : f32
  %empty3 = tensor.empty() : tensor<8x8xf32>
  %init3 = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduced = linalg.reduce ins(%mul : tensor<8x8x16xf32>)
                           outs(%init3 : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  return %reduced : tensor<8x8xf32>
}

// Single additive reduce with output 8x8, tile-m=4 tile-n=4:
// M=8 > 4 → tiled, N=8 > 4 → tiled. Verify scf.for appears.
// CHECK: scf.for
// Verify fused producers inside the loop body.
// CHECK:   linalg.generic
// CHECK:   linalg.reduce
// CHECK:     arith.addf


// ============================================================================
// Test 3: Non-additive reduce — NOT identified as root, IR unchanged
// ============================================================================
// A reduce with arith.maxnumf combiner should NOT be identified as a reduce
// chain root. The pass should leave the IR unchanged (Requirement 1.6).

// CHECK-LABEL: func.func @non_additive_reduce
func.func @non_additive_reduce(%arg0: tensor<8x1x16xf32>,
                                %arg1: tensor<1x8x16xf32>) -> tensor<8x8xf32> {
  // Broadcast %arg0 on dim 1
  %empty0 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1_3, #map_identity3b],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<8x1x16xf32>) outs(%empty0 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Broadcast %arg1 on dim 0
  %empty1 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0_3, #map_identity3b],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x16xf32>) outs(%empty1 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Element-wise multiply
  %empty2 = tensor.empty() : tensor<8x8x16xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity3b, #map_identity3b, #map_identity3b],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<8x8x16xf32>, tensor<8x8x16xf32>)
    outs(%empty2 : tensor<8x8x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x16xf32>

  // Non-additive reduce (maxnumf combiner) — should NOT be a root
  %neg_inf = arith.constant 0xFF800000 : f32
  %empty3 = tensor.empty() : tensor<8x8xf32>
  %init3 = linalg.fill ins(%neg_inf : f32) outs(%empty3 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduced = linalg.reduce ins(%mul : tensor<8x8x16xf32>)
                           outs(%init3 : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %max = arith.maxnumf %in, %acc : f32
      linalg.yield %max : f32
    }

  return %reduced : tensor<8x8xf32>
}

// Non-additive reduce should NOT be identified as root → no tiling.
// CHECK-NOT: scf.for
// Verify the original reduce is still present unchanged.
// CHECK: linalg.reduce
// CHECK:   arith.maxnumf
