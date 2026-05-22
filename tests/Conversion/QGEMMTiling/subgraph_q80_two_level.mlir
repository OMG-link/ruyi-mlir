// RUN: buddy-opt --qgemm-tiling="tile-m=32 tile-n=32" --debug-only=qgemm-tiling %s 2>&1 | FileCheck %s
// REQUIRES: asserts

// Verify subgraph collection on a two-level reduction pattern (closer to real
// Q8_0 × Q8_0). The pattern has:
// - 4D broadcast generics (broadcast on dim 0 and dim 1)
// - 4D element-wise multiply
// - Inner reduce (dim=3, arith.addi)
// - 3D element-wise scale multiply with broadcast scale factors
// - Outer reduce (dim=2, arith.addf) — this is the tiling root
//
// Expected: hasBroadcast = [true, true], reduces = 2

// CHECK: Reduce chain root: {{.*}}linalg.reduce
// CHECK: Generic (continue):
// CHECK: Broadcast detected: iter dim 1 -> output dim 1
// CHECK: Generic (continue):
// CHECK: Broadcast detected: iter dim 0 -> output dim 0
// CHECK: Generic (continue):
// CHECK: Reduce (continue): {{.*}}linalg.reduce
// CHECK: ContractionSubgraph collected:
// CHECK-NEXT: ops: 11
// CHECK-NEXT: reduces: 2
// CHECK-NEXT: hasBroadcast: [true, true]
// CHECK-LABEL: func.func @q80_two_level

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_bcast4_d1 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map_bcast4_d0 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast3_d1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map_bcast3_d0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>

func.func @q80_two_level(%a_q: tensor<4x1x2x8xi16>, %w_q: tensor<1x8x2x8xi16>,
                          %a_d: tensor<4x1x2xf32>, %w_d: tensor<1x8x2xf32>,
                          %out: memref<4x8xf32>) {
  // --- 4D broadcast + multiply + inner reduce (dim=3) ---

  // Broadcast a_q on dim 1
  %empty_4d = tensor.empty() : tensor<4x8x2x8xi16>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast4_d1, #map4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%a_q : tensor<4x1x2x8xi16>) outs(%empty_4d : tensor<4x8x2x8xi16>) {
  ^bb0(%in: i16, %out_e: i16):
    linalg.yield %in : i16
  } -> tensor<4x8x2x8xi16>

  // Broadcast w_q on dim 0
  %empty_4d_1 = tensor.empty() : tensor<4x8x2x8xi16>
  %bcast_w = linalg.generic {
    indexing_maps = [#map_bcast4_d0, #map4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%w_q : tensor<1x8x2x8xi16>) outs(%empty_4d_1 : tensor<4x8x2x8xi16>) {
  ^bb0(%in: i16, %out_e: i16):
    linalg.yield %in : i16
  } -> tensor<4x8x2x8xi16>

  // Element-wise multiply (4D)
  %empty_4d_2 = tensor.empty() : tensor<4x8x2x8xi32>
  %mul = linalg.generic {
    indexing_maps = [#map4, #map4, #map4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_w : tensor<4x8x2x8xi16>, tensor<4x8x2x8xi16>)
    outs(%empty_4d_2 : tensor<4x8x2x8xi32>) {
  ^bb0(%in0: i16, %in1: i16, %out_e: i32):
    %a = arith.extsi %in0 : i16 to i32
    %b = arith.extsi %in1 : i16 to i32
    %p = arith.muli %a, %b : i32
    linalg.yield %p : i32
  } -> tensor<4x8x2x8xi32>

  // Inner reduce (dim=3, arith.addi)
  %c0_i32 = arith.constant 0 : i32
  %empty_3d_init = tensor.empty() : tensor<4x8x2xi32>
  %fill_inner = linalg.fill ins(%c0_i32 : i32) outs(%empty_3d_init : tensor<4x8x2xi32>) -> tensor<4x8x2xi32>
  %inner_reduced = linalg.reduce ins(%mul : tensor<4x8x2x8xi32>)
                                 outs(%fill_inner : tensor<4x8x2xi32>)
                                 dimensions = [3]
    (%in: i32, %acc: i32) {
      %s = arith.addi %in, %acc : i32
      linalg.yield %s : i32
    }

  // --- 3D scale multiply + outer reduce (dim=2) ---

  // Cast inner reduce result to f32
  %empty_3d_f32 = tensor.empty() : tensor<4x8x2xf32>
  %cast = linalg.generic {
    indexing_maps = [#map3, #map3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner_reduced : tensor<4x8x2xi32>) outs(%empty_3d_f32 : tensor<4x8x2xf32>) {
  ^bb0(%in: i32, %out_e: f32):
    %f = arith.sitofp %in : i32 to f32
    linalg.yield %f : f32
  } -> tensor<4x8x2xf32>

  // Broadcast a_d on dim 1
  %empty_3d_scale = tensor.empty() : tensor<4x8x2xf32>
  %bcast_ad = linalg.generic {
    indexing_maps = [#map_bcast3_d1, #map3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%a_d : tensor<4x1x2xf32>) outs(%empty_3d_scale : tensor<4x8x2xf32>) {
  ^bb0(%in: f32, %out_e: f32):
    linalg.yield %in : f32
  } -> tensor<4x8x2xf32>

  // Broadcast w_d on dim 0
  %empty_3d_scale2 = tensor.empty() : tensor<4x8x2xf32>
  %bcast_wd = linalg.generic {
    indexing_maps = [#map_bcast3_d0, #map3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w_d : tensor<1x8x2xf32>) outs(%empty_3d_scale2 : tensor<4x8x2xf32>) {
  ^bb0(%in: f32, %out_e: f32):
    linalg.yield %in : f32
  } -> tensor<4x8x2xf32>

  // Scale multiply: cast * a_d * w_d
  %empty_3d_scaled = tensor.empty() : tensor<4x8x2xf32>
  %scaled = linalg.generic {
    indexing_maps = [#map3, #map3, #map3, #map3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%cast, %bcast_ad, %bcast_wd : tensor<4x8x2xf32>, tensor<4x8x2xf32>, tensor<4x8x2xf32>)
    outs(%empty_3d_scaled : tensor<4x8x2xf32>) {
  ^bb0(%in0: f32, %in1: f32, %in2: f32, %out_e: f32):
    %t = arith.mulf %in0, %in1 : f32
    %r = arith.mulf %t, %in2 : f32
    linalg.yield %r : f32
  } -> tensor<4x8x2xf32>

  // Outer reduce (dim=2, arith.addf) — TILING ROOT
  %cst = arith.constant 0.0 : f32
  %empty_2d = tensor.empty() : tensor<4x8xf32>
  %fill_outer = linalg.fill ins(%cst : f32) outs(%empty_2d : tensor<4x8xf32>) -> tensor<4x8xf32>
  %outer_reduced = linalg.reduce ins(%scaled : tensor<4x8x2xf32>)
                                 outs(%fill_outer : tensor<4x8xf32>)
                                 dimensions = [2]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }

  // Store result
  bufferization.materialize_in_destination %outer_reduced in writable %out
    : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
  return
}
