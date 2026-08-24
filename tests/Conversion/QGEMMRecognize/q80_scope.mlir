// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s

// The recognizer records the complete root-owned scope and leaves the
// root-neutral fill/empty boundaries unannotated.
//
// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] op 0 walk: ops=4 boundaries=6
// CHECK: [qgemm-recognize] op 0 accepted: root=0
// CHECK-LABEL: func.func @q80_root_scope
// CHECK: qgemm.node = {root = 0 : i64, stage = "compute"}
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

#map_identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d1  = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map_bcast_d0  = affine_map<(d0, d1, d2) -> (0, d1, d2)>

func.func @q80_root_scope(%arg0: tensor<4x1x8xf32>, %arg1: tensor<1x8x8xf32>,
                         %out: memref<4x8xf32>) {
  // Broadcast %arg0 on dim 1: (d0,d1,d2) -> (d0, 0, d2)
  %empty0 = tensor.empty() : tensor<4x8x8xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<4x1x8xf32>) outs(%empty0 : tensor<4x8x8xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<4x8x8xf32>

  // Broadcast %arg1 on dim 0: (d0,d1,d2) -> (0, d1, d2)
  %empty1 = tensor.empty() : tensor<4x8x8xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x8xf32>) outs(%empty1 : tensor<4x8x8xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<4x8x8xf32>

  // Element-wise multiply
  %empty2 = tensor.empty() : tensor<4x8x8xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity3, #map_identity3, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<4x8x8xf32>, tensor<4x8x8xf32>)
    outs(%empty2 : tensor<4x8x8xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<4x8x8xf32>

  // Reduce on dim 2 (additive combiner)
  %cst = arith.constant 0.0 : f32
  %empty3 = tensor.empty() : tensor<4x8xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<4x8xf32>) -> tensor<4x8xf32>
  %reduced = linalg.reduce ins(%mul : tensor<4x8x8xf32>)
                           outs(%init : tensor<4x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Store result
  bufferization.materialize_in_destination %reduced in writable %out
    : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
  return
}
