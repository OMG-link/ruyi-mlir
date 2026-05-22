// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s | FileCheck %s
//
// Tests for pass-internal DCE cleanup (Requirements 7.3, 7.4, 7.5):
//   1. After tiling, original untiled ops (cloned into the loop body) are
//      removed by DCE because they have no remaining users.
//   2. Ops that still have live users (e.g., consumed by
//      bufferization.materialize_in_destination) are preserved.
//   3. The resulting IR passes the MLIR verifier without errors.
//
// Test scenario: A contraction pattern (broadcast → multiply → reduce) whose
// result feeds into bufferization.materialize_in_destination. After tiling:
//   - The tiled result (from scf.for) feeds materialize_in_destination.
//   - The original untiled broadcast, multiply, and reduce ops are dead
//     (their uses were replaced by the tiled versions).
//   - DCE removes the dead original ops.
//   - materialize_in_destination is preserved (live consumer of tiled result).

#map_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map_bcast_d1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

// CHECK-LABEL: func.func @dce_removes_dead_ops
func.func @dce_removes_dead_ops(%arg0: tensor<8x1x16xf32>,
                                 %arg1: tensor<1x8x16xf32>,
                                 %out: memref<8x8xf32>) {
  %cst = arith.constant 0.0 : f32

  // Broadcast %arg0 on dim 1: (8,1,16) → (8,8,16)
  %empty0 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<8x1x16xf32>) outs(%empty0 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Broadcast %arg1 on dim 0: (1,8,16) → (8,8,16)
  %empty1 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x16xf32>) outs(%empty1 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Element-wise multiply
  %empty2 = tensor.empty() : tensor<8x8x16xf32>
  %mul = linalg.generic {
    indexing_maps = [#map_identity, #map_identity, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<8x8x16xf32>, tensor<8x8x16xf32>)
    outs(%empty2 : tensor<8x8x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x16xf32>

  // Additive reduce on dim 2 — tiling root
  %empty3 = tensor.empty() : tensor<8x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduced = linalg.reduce ins(%mul : tensor<8x8x16xf32>)
                           outs(%fill : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // Store result — this is the live consumer that must be preserved.
  bufferization.materialize_in_destination %reduced in writable %out
    : (tensor<8x8xf32>, memref<8x8xf32>) -> ()
  return
}

// After tiling with tile-m=4 tile-n=4 on output shape 8x8:
//   M=8 > 4 → tiled, N=8 > 4 → tiled.
// The tiled result feeds materialize_in_destination.
// Original untiled linalg ops are dead and removed by DCE.

// Verify tiling happened — nested scf.for loops are present.
// CHECK: scf.for
// CHECK:   scf.for

// Verify fused producers inside the tiled loop body.
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.fill
// CHECK:     linalg.reduce
// CHECK:       arith.addf

// Verify materialize_in_destination is preserved (live consumer of tiled result).
// CHECK: bufferization.materialize_in_destination

// Verify dead original untiled ops are removed by DCE:
// The original top-level linalg.generic ops (broadcasts, multiply) and
// linalg.reduce that operated on the full untiled tensors should be gone.
// Only linalg.generic/reduce ops inside the scf.for body should remain.
// After materialize_in_destination, there should be no more linalg ops
// before return — the dead originals have been cleaned up.
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.reduce
// CHECK: return
