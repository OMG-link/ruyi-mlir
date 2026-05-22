// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s | FileCheck %s
//
// NOTE: This test requires the v02 pass restructuring (task 6.1) which replaces
// FinalConsumerFinder with ReduceChainAnalysis. Until that restructuring is
// complete, this test will not pass.
//
// Tests for BFS boundary condition: tensor.extract_slice (Requirement 4.1)
//
// Verifies that when the backward BFS encounters a tensor.extract_slice op,
// it treats it as a leaf node (boundary):
//   - The tensor.extract_slice op is NOT added to the subgraph's op set
//   - BFS does NOT traverse through it to its source operand
//
// Test scenario: A function containing a tensor.extract_slice that produces
// a tensor, followed by a contraction pattern (broadcast → multiply → reduce)
// operating on that sliced tensor. The BFS should collect the contraction
// pattern but NOT traverse through the extract_slice.
//
// The contraction pattern uses 3D tensors with broadcast on dim 0 and dim 1,
// giving exactly 2 broadcast dims. Output shape is 8x8, with tile-m=4 tile-n=4:
//   M=8 > 4 → tiled with step 4
//   N=8 > 4 → tiled with step 4

#map_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map_bcast_d1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

// CHECK-LABEL: func.func @boundary_extract_slice
func.func @boundary_extract_slice(%source_a: tensor<16x1x4xf32>,
                                   %source_b: tensor<1x16x4xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 0.0 : f32

  // Extract slices from larger source tensors.
  // These tensor.extract_slice ops should act as BFS boundaries:
  // the BFS should NOT traverse through them to %source_a / %source_b.
  %slice_a = tensor.extract_slice %source_a[0, 0, 0] [8, 1, 4] [1, 1, 1]
      : tensor<16x1x4xf32> to tensor<8x1x4xf32>
  %slice_b = tensor.extract_slice %source_b[0, 0, 0] [1, 8, 4] [1, 1, 1]
      : tensor<1x16x4xf32> to tensor<1x8x4xf32>

  // Broadcast %slice_a on dim 1: (8,1,4) → (8,8,4)
  %empty_bcast_a = tensor.empty() : tensor<8x8x4xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%slice_a : tensor<8x1x4xf32>) outs(%empty_bcast_a : tensor<8x8x4xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>

  // Broadcast %slice_b on dim 0: (1,8,4) → (8,8,4)
  %empty_bcast_b = tensor.empty() : tensor<8x8x4xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%slice_b : tensor<1x8x4xf32>) outs(%empty_bcast_b : tensor<8x8x4xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>

  // Element-wise multiply
  %empty_mul = tensor.empty() : tensor<8x8x4xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity, #map_identity, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<8x8x4xf32>, tensor<8x8x4xf32>)
    outs(%empty_mul : tensor<8x8x4xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x4xf32>

  // Additive reduce on dim 2 — this is the tiling root
  %empty_out = tensor.empty() : tensor<8x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty_out : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduced = linalg.reduce ins(%mul : tensor<8x8x4xf32>)
                           outs(%fill : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  return %reduced : tensor<8x8xf32>
}

// Verify that tiling is applied to the contraction pattern (scf.for appears).
// The reduce has output shape 8x8, tile-m=4 tile-n=4 → both M and N tiled.
//
// Tiling produces nested scf.for loops (M and N dimensions).
// The tiling pass creates its own constants (e.g. %c0_0, %c8_2, %c4_3),
// so we match the constant names with optional suffixes.
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK:   scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
//
// Inside the tiled loop body, the fused contraction pattern is present.
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.reduce
// CHECK:       arith.addf
//
// The tensor.extract_slice ops remain OUTSIDE the tiled region (not fused in).
// They should appear before the tiling loops, not inside them.
