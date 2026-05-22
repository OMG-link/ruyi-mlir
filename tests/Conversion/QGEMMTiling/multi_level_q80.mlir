// RUN: buddy-opt --qgemm-tiling="tile-m=32 tile-n=32" --qgemm-tiling="tile-m=4 tile-n=4" %s | FileCheck %s
//
// Multi-level tiling test for Q8_0 pattern (Requirements 8.1, 8.2, 9.1, 9.2).
//
// The Q8_0 pattern has output shape tensor<64x32xf32>.
// First tiling (tile-m=32, tile-n=32):
//   M=64 > 32 → outer scf.for over M (0 to 64 step 32)
//   N=32 ≤ 32 → no N loop
// Second tiling (tile-m=4, tile-n=4):
//   M=32 > 4 → inner scf.for over M (0 to 32 step 4)
//   N=32 > 4 → inner scf.for over N (0 to 32 step 4)
//
// Verifies:
//   1. Outer scf.for with step 32 (M dimension)
//   2. Inner scf.for loops with step 4 (M and N dimensions)
//   3. Producers (broadcast, multiply, reduce) fused into innermost loop body

#map_identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_bcast_d0  = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map_bcast_d1  = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>

func.func @multi_level_q80(%arg0: tensor<64x1x2x32xf32>, %arg1: tensor<1x32x2x32xf32>,
                            %out: memref<64x32xf32>) {
  // Broadcast %arg0 on dim 1: input has shape (64, 1, 2, 32)
  %empty0 = tensor.empty() : tensor<64x32x2x32xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<64x1x2x32xf32>) outs(%empty0 : tensor<64x32x2x32xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<64x32x2x32xf32>

  // Broadcast %arg1 on dim 0: input has shape (1, 32, 2, 32)
  %empty1 = tensor.empty() : tensor<64x32x2x32xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x32x2x32xf32>) outs(%empty1 : tensor<64x32x2x32xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<64x32x2x32xf32>

  // Element-wise multiply (4D)
  %empty2 = tensor.empty() : tensor<64x32x2x32xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity4, #map_identity4, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<64x32x2x32xf32>, tensor<64x32x2x32xf32>)
    outs(%empty2 : tensor<64x32x2x32xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<64x32x2x32xf32>

  // Reduce on dim 3 (QK dimension, additive combiner)
  %cst = arith.constant 0.0 : f32
  %empty3 = tensor.empty() : tensor<64x32x2xf32>
  %init3 = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<64x32x2xf32>) -> tensor<64x32x2xf32>
  %reduced3 = linalg.reduce ins(%mul : tensor<64x32x2x32xf32>)
                            outs(%init3 : tensor<64x32x2xf32>)
                            dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Reduce on dim 2 (KB dimension, additive combiner) — this is the tiling root
  %empty4 = tensor.empty() : tensor<64x32xf32>
  %init4 = linalg.fill ins(%cst : f32) outs(%empty4 : tensor<64x32xf32>) -> tensor<64x32xf32>
  %reduced2 = linalg.reduce ins(%reduced3 : tensor<64x32x2xf32>)
                            outs(%init4 : tensor<64x32xf32>)
                            dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Store result
  bufferization.materialize_in_destination %reduced2 in writable %out
    : (tensor<64x32xf32>, memref<64x32xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @multi_level_q80

// Outer M-dimension tiling loop: from 0 to 64 step 32
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c64{{[_0-9]*}} step %c32{{[_0-9]*}}

// Inner M-dimension tiling loop: from 0 to 32 step 4
// CHECK:   scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}

// Inner N-dimension tiling loop: from 0 to 32 step 4
// CHECK:     scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}

// Verify all producers fused into innermost loop body:
// broadcast A, broadcast B, element-wise multiply
// CHECK:       linalg.generic
// CHECK:       linalg.generic
// CHECK:       linalg.generic

// Reduce on dim 3 (QK reduction) fused into innermost body
// CHECK:       linalg.reduce
// CHECK:         arith.addf

// Reduce on dim 2 (KB reduction) — tiling root — in innermost body
// CHECK:       linalg.reduce
// CHECK:         arith.addf

// Verify materialize_in_destination still exists after all loops
// CHECK: bufferization.materialize_in_destination
