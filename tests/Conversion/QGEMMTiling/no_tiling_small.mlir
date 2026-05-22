// RUN: buddy-opt --qgemm-tiling="tile-m=32 tile-n=32" %s | FileCheck %s

// Verify that when all output dimensions are ≤ the tile sizes, the pass
// does NOT tile (IR is unchanged). With output shape 16x16, tile-m=32,
// tile-n=32: M=16 ≤ 32 → tile=0, N=16 ≤ 32 → tile=0, all zero → early return.

#map_identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_bcast_d0  = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map_bcast_d1  = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>

func.func @q80_no_tiling_small(%arg0: tensor<16x1x2x32xf32>, %arg1: tensor<1x16x2x32xf32>,
                                %out: memref<16x16xf32>) {
  // Broadcast %arg0 on dim 1
  %empty0 = tensor.empty() : tensor<16x16x2x32xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<16x1x2x32xf32>) outs(%empty0 : tensor<16x16x2x32xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<16x16x2x32xf32>

  // Broadcast %arg1 on dim 0
  %empty1 = tensor.empty() : tensor<16x16x2x32xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x16x2x32xf32>) outs(%empty1 : tensor<16x16x2x32xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<16x16x2x32xf32>

  // Element-wise multiply
  %empty2 = tensor.empty() : tensor<16x16x2x32xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity4, #map_identity4, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<16x16x2x32xf32>, tensor<16x16x2x32xf32>)
    outs(%empty2 : tensor<16x16x2x32xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<16x16x2x32xf32>

  // Reduce on dim 3 (QK dimension)
  %cst = arith.constant 0.0 : f32
  %empty3 = tensor.empty() : tensor<16x16x2xf32>
  %init3 = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<16x16x2xf32>) -> tensor<16x16x2xf32>
  %reduced3 = linalg.reduce ins(%mul : tensor<16x16x2x32xf32>)
                            outs(%init3 : tensor<16x16x2xf32>)
                            dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Reduce on dim 2 (KB dimension) — this is the tiling root
  %empty4 = tensor.empty() : tensor<16x16xf32>
  %init4 = linalg.fill ins(%cst : f32) outs(%empty4 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reduced2 = linalg.reduce ins(%reduced3 : tensor<16x16x2xf32>)
                            outs(%init4 : tensor<16x16xf32>)
                            dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Store result
  bufferization.materialize_in_destination %reduced2 in writable %out
    : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  return
}

// Verify no tiling loop is generated (all dims ≤ tile sizes → IR unchanged)
// CHECK-LABEL: func.func @q80_no_tiling_small
// CHECK-NOT: scf.for

// Verify original reduces still present
// CHECK: linalg.reduce
// CHECK: linalg.reduce

// Verify store still present
// CHECK: bufferization.materialize_in_destination
