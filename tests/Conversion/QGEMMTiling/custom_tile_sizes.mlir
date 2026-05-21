// RUN: buddy-opt --qgemm-tiling="tile-m=16 tile-n=8" %s | FileCheck %s

// Verify that the QGEMM tiling pass generates scf.for loops with correct
// bounds for non-default tile sizes. With output shape 128x64, tile-m=16,
// tile-n=8:
//   M=128 tiled by 16 → loop from 0 to 128 step 16
//   N=64 tiled by 8  → loop from 0 to 64 step 8
// Both M and N are tiled, producing two nested scf.for loops.

#map_identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_bcast_d0  = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map_bcast_d1  = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>

func.func @custom_tile_sizes(%arg0: tensor<128x1x2x32xf32>, %arg1: tensor<1x64x2x32xf32>,
                              %out: memref<128x64xf32>) {
  // Broadcast %arg0 on dim 1: input has shape (128, 1, 2, 32)
  %empty0 = tensor.empty() : tensor<128x64x2x32xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<128x1x2x32xf32>) outs(%empty0 : tensor<128x64x2x32xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<128x64x2x32xf32>

  // Broadcast %arg1 on dim 0: input has shape (1, 64, 2, 32)
  %empty1 = tensor.empty() : tensor<128x64x2x32xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x64x2x32xf32>) outs(%empty1 : tensor<128x64x2x32xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<128x64x2x32xf32>

  // Element-wise multiply (4D)
  %empty2 = tensor.empty() : tensor<128x64x2x32xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity4, #map_identity4, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<128x64x2x32xf32>, tensor<128x64x2x32xf32>)
    outs(%empty2 : tensor<128x64x2x32xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<128x64x2x32xf32>

  // Reduce on dim 3 (QK dimension, additive combiner)
  %cst = arith.constant 0.0 : f32
  %empty3 = tensor.empty() : tensor<128x64x2xf32>
  %init3 = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<128x64x2xf32>) -> tensor<128x64x2xf32>
  %reduced3 = linalg.reduce ins(%mul : tensor<128x64x2x32xf32>)
                            outs(%init3 : tensor<128x64x2xf32>)
                            dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Reduce on dim 2 (KB dimension, additive combiner) — this is the tiling root
  %empty4 = tensor.empty() : tensor<128x64xf32>
  %init4 = linalg.fill ins(%cst : f32) outs(%empty4 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %reduced2 = linalg.reduce ins(%reduced3 : tensor<128x64x2xf32>)
                            outs(%init4 : tensor<128x64xf32>)
                            dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Store result
  bufferization.materialize_in_destination %reduced2 in writable %out
    : (tensor<128x64xf32>, memref<128x64xf32>) -> ()
  return
}

// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index

// Verify two nested scf.for loops with correct bounds
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C128]] step %[[C16]]
// CHECK:   scf.for %{{.*}} = %{{.*}} to %[[C64]] step %[[C8]]

// Verify fused producers inside the inner loop body
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.reduce
// CHECK:       arith.addf
// CHECK:     linalg.reduce
// CHECK:       arith.addf

// Verify the materialize_in_destination still exists (after the loops)
// CHECK: bufferization.materialize_in_destination
