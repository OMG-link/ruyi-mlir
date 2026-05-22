// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s 2>&1 | FileCheck %s
//
// NOTE: This test requires the v02 pass restructuring (task 6.1) which uses
// RootFilter for diamond detection. Until that restructuring is complete,
// this test will not pass.
//
// Tests for diamond detection and skip (Requirement 2.4):
//   Two independent reduce chains share a common element-wise producer op
//   (diamond pattern). Both roots should be skipped with a warning, and the
//   IR should remain unchanged.

// ============================================================================
// Test: Diamond pattern — two reduce chains sharing a producer op
// ============================================================================
// Pattern:
//   %shared_mul = linalg.generic (element-wise multiply)
//       ↙                    ↘
//   %reduce1 (addf, dim=2)    %reduce2 (addf, dim=2)
//
// Both reduce chains include %shared_mul in their subgraphs. The RootFilter
// detects this overlap (refcount > 1) and skips both roots with a warning.

#map_identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d1  = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map_bcast_d0  = affine_map<(d0, d1, d2) -> (0, d1, d2)>

func.func @diamond_two_roots(%arg0: tensor<8x1x16xf32>,
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

  // Shared element-wise multiply — this is the diamond junction point.
  // Both reduce chains below include this op in their subgraphs.
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

// Both roots share %shared_mul → diamond detected → both skipped.
// Verify the warning is emitted (at least once, both roots trigger it).
// The warning appears in stderr (merged via 2>&1) before the IR output,
// so it must be checked before the CHECK-LABEL.
// CHECK: warning: qgemm-tiling: subgraph shares ops with another reduce chain root (diamond); skipping

// CHECK-LABEL: func.func @diamond_two_roots

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
