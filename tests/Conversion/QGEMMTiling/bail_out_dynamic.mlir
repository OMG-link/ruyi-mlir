// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32 explain" %s 2>&1 | FileCheck %s --check-prefix=EXPLAIN
// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" %s 2>&1 | FileCheck %s --check-prefix=SILENT

// Verify that when the M dimension is dynamic, the pass emits a remark
// and leaves the IR unchanged (no tiling loops generated).

#map_identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_bcast_d0  = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map_bcast_d1  = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>

func.func @dynamic_m(%arg0: tensor<?x1x2x32xf32>, %arg1: tensor<1x32x2x32xf32>,
                      %out: memref<?x32xf32>) {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x1x2x32xf32>

  // Broadcast %arg0 on dim 1: input has shape (?, 1, 2, 32)
  %empty0 = tensor.empty(%dim) : tensor<?x32x2x32xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<?x1x2x32xf32>) outs(%empty0 : tensor<?x32x2x32xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<?x32x2x32xf32>

  // Broadcast %arg1 on dim 0: input has shape (1, 32, 2, 32)
  %empty1 = tensor.empty(%dim) : tensor<?x32x2x32xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x32x2x32xf32>) outs(%empty1 : tensor<?x32x2x32xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<?x32x2x32xf32>

  // Element-wise multiply (4D)
  %empty2 = tensor.empty(%dim) : tensor<?x32x2x32xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity4, #map_identity4, #map_identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<?x32x2x32xf32>, tensor<?x32x2x32xf32>)
    outs(%empty2 : tensor<?x32x2x32xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<?x32x2x32xf32>

  // Reduce on dim 3 (QK dimension, additive combiner)
  %cst = arith.constant 0.0 : f32
  %empty3 = tensor.empty(%dim) : tensor<?x32x2xf32>
  %init3 = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<?x32x2xf32>) -> tensor<?x32x2xf32>
  %reduced3 = linalg.reduce ins(%mul : tensor<?x32x2x32xf32>)
                            outs(%init3 : tensor<?x32x2xf32>)
                            dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Reduce on dim 2 (KB dimension, additive combiner) — this is the tiling root
  %empty4 = tensor.empty(%dim) : tensor<?x32xf32>
  %init4 = linalg.fill ins(%cst : f32) outs(%empty4 : tensor<?x32xf32>) -> tensor<?x32xf32>
  %reduced2 = linalg.reduce ins(%reduced3 : tensor<?x32x2xf32>)
                            outs(%init4 : tensor<?x32xf32>)
                            dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Store result
  bufferization.materialize_in_destination %reduced2 in writable %out
    : (tensor<?x32xf32>, memref<?x32xf32>) -> ()
  return
}

// EXPLAIN: remark{{.*}}dynamic M/N dimension in tiling root; skipping
// EXPLAIN-LABEL: func.func @dynamic_m
// EXPLAIN-NOT: scf.for
// EXPLAIN: linalg.reduce
// SILENT-NOT: remark
// SILENT-LABEL: func.func @dynamic_m
// SILENT-NOT: scf.for
// SILENT: linalg.reduce
