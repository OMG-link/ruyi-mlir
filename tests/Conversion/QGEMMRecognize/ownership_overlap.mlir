// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s

// Two candidates share one root-owned producer. Recognition rejects both by
// the cross-candidate ownership check, without emitting tiling diagnostics.
// CHECK: [qgemm-recognize] candidates: 2
// CHECK: [qgemm-recognize] op 0 walk: ops=4 boundaries=6
// CHECK: [qgemm-recognize] op 1 walk: ops=4 boundaries=6
// CHECK: [qgemm-recognize] candidate 0 rejected: reason=shared-non-boundary-operation section=13 at=linalg.generic
// CHECK: [qgemm-recognize] candidate 1 rejected: reason=shared-non-boundary-operation section=13 at=linalg.generic
//
// Two independent candidates share a common element-wise producer. The
// cross-candidate ownership check rejects both roots and leaves the IR
// unannotated.

// ============================================================================
// Test: two roots sharing one root-owned producer
// ============================================================================
// Pattern:
//   %shared_mul = linalg.generic (element-wise multiply)
//       ↙                    ↘
//   %reduce1 (addf, dim=2)    %reduce2 (addf, dim=2)
//
// Both candidate walks include %shared_mul. The ownership check rejects both
// candidates because accepted root-owned scopes must be disjoint.

#map_identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d1  = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map_bcast_d0  = affine_map<(d0, d1, d2) -> (0, d1, d2)>

func.func @ownership_overlap_two_roots(%arg0: tensor<8x1x16xf32>,
                              %arg1: tensor<1x8x16xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // Broadcast %arg0 on dim 1
  %empty0 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<8x1x16xf32>) outs(%empty0 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Broadcast %arg1 on dim 0
  %empty1 = tensor.empty() : tensor<8x8x16xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x16xf32>) outs(%empty1 : tensor<8x8x16xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x16xf32>

  // Shared element-wise multiply — this is the shared-producer junction.
  // Both candidate walks include this op in their root-owned scopes.
  %empty2 = tensor.empty() : tensor<8x8x16xf32>
  %shared_mul = linalg.generic {
    indexing_maps = [#map_identity3, #map_identity3, #map_identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<8x8x16xf32>, tensor<8x8x16xf32>)
    outs(%empty2 : tensor<8x8x16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x16xf32>

  // --- Reduce chain 1: reduce %shared_mul on dim 2 ---
  %cst = arith.constant 0.0 : f32
  %empty3 = tensor.empty() : tensor<8x8xf32>
  %init1 = linalg.fill ins(%cst : f32) outs(%empty3 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduce1 = linalg.reduce ins(%shared_mul : tensor<8x8x16xf32>)
                           outs(%init1 : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // --- Reduce chain 2: reduce %shared_mul on dim 2 (independent root) ---
  %empty4 = tensor.empty() : tensor<8x8xf32>
  %init2 = linalg.fill ins(%cst : f32) outs(%empty4 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %reduce2 = linalg.reduce ins(%shared_mul : tensor<8x8x16xf32>)
                           outs(%init2 : tensor<8x8xf32>)
                           dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  return %reduce1, %reduce2 : tensor<8x8xf32>, tensor<8x8xf32>
}

// Both roots share %shared_mul → shared-producer overlap detected → both skipped.
// Verify the warning is emitted (at least once, both roots trigger it).
// The warning appears in stderr (merged via 2>&1) before the IR output,
// so it must be checked before the CHECK-LABEL.
// The IR remains unannotated after the ownership rejection.

// CHECK-LABEL: func.func @ownership_overlap_two_roots
// CHECK-NOT: qgemm.node
// CHECK-NOT: qgemm.root

// Verify IR is unchanged — no scf.for introduced (no tiling happened).
// CHECK-NOT: scf.for

// Verify the original ops are still present.
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.reduce
// CHECK:   arith.addf
// CHECK: linalg.reduce
// CHECK:   arith.addf
