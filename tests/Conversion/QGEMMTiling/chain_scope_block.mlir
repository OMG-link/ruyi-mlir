// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s | FileCheck %s

// Verify that reduce chain analysis scopes chains to basic blocks (Req 1.7)
// and searches for roots at all nesting levels (Req 1.8).
//
// This test has:
//   1. An scf.for loop containing a linalg.reduce (additive) inside its body
//   2. Another linalg.reduce (additive) outside the loop at the function level
//   3. Even though the inner reduce's result feeds back through iter_args,
//      the two reduces are in different blocks and must be independent roots
//
// Expected behavior:
//   - The inner reduce (inside scf.for body) is an independent root in its block
//   - The outer reduce (at function level) is an independent root in its block
//   - They are NOT considered part of the same chain (different blocks)
//   - Both are eligible for tiling independently
//
// Both contraction patterns use 3D tensors with one broadcast dimension each
// (broadcast on dim 0 and broadcast on dim 1), giving exactly 2 broadcast dims.
// Output shape is 8x8, with tile-m=4, tile-n=4:
//   M=8 > 4 → tiled with step 4
//   N=8 > 4 → tiled with step 4

#map_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map_bcast_d1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

func.func @chain_scope_block(%a_outer: tensor<8x1x4xf32>,
                              %b_outer: tensor<1x8x4xf32>,
                              %a_inner: tensor<8x1x4xf32>,
                              %b_inner: tensor<1x8x4xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index

  // ===== Outer contraction pattern (at function level) =====

  // Broadcast a_outer on dim 1: (8,1,4) → (8,8,4)
  %empty_bcast_a = tensor.empty() : tensor<8x8x4xf32>
  %bcast_a = linalg.generic {
    indexing_maps = [#map_bcast_d1, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%a_outer : tensor<8x1x4xf32>) outs(%empty_bcast_a : tensor<8x8x4xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>

  // Broadcast b_outer on dim 0: (1,8,4) → (8,8,4)
  %empty_bcast_b = tensor.empty() : tensor<8x8x4xf32>
  %bcast_b = linalg.generic {
    indexing_maps = [#map_bcast_d0, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%b_outer : tensor<1x8x4xf32>) outs(%empty_bcast_b : tensor<8x8x4xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>

  // Element-wise multiply
  %empty_mul = tensor.empty() : tensor<8x8x4xf32>
  %mul_outer = linalg.generic {
    indexing_maps = [#map_identity, #map_identity, #map_identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bcast_a, %bcast_b : tensor<8x8x4xf32>, tensor<8x8x4xf32>)
    outs(%empty_mul : tensor<8x8x4xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %prod = arith.mulf %in0, %in1 : f32
    linalg.yield %prod : f32
  } -> tensor<8x8x4xf32>

  // Outer reduce (dim=2, additive) — ROOT at function level
  %empty_out = tensor.empty() : tensor<8x8xf32>
  %fill_outer = linalg.fill ins(%cst : f32) outs(%empty_out : tensor<8x8xf32>) -> tensor<8x8xf32>
  %outer_reduced = linalg.reduce ins(%mul_outer : tensor<8x8x4xf32>)
                                 outs(%fill_outer : tensor<8x8xf32>)
                                 dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  // ===== scf.for loop with inner contraction pattern =====
  // The loop iterates and accumulates into a tensor<8x8xf32>.
  // Inside the loop body, there is an independent contraction pattern.
  %fill_loop = linalg.fill ins(%cst : f32) outs(%empty_out : tensor<8x8xf32>) -> tensor<8x8xf32>
  %loop_result = scf.for %iv = %c0 to %c3 step %c1
      iter_args(%acc = %fill_loop) -> (tensor<8x8xf32>) {

    // Broadcast a_inner on dim 1: (8,1,4) → (8,8,4)
    %empty_inner_bcast_a = tensor.empty() : tensor<8x8x4xf32>
    %inner_bcast_a = linalg.generic {
      indexing_maps = [#map_bcast_d1, #map_identity],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a_inner : tensor<8x1x4xf32>) outs(%empty_inner_bcast_a : tensor<8x8x4xf32>) {
    ^bb0(%in: f32, %out_elem: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x4xf32>

    // Broadcast b_inner on dim 0: (1,8,4) → (8,8,4)
    %empty_inner_bcast_b = tensor.empty() : tensor<8x8x4xf32>
    %inner_bcast_b = linalg.generic {
      indexing_maps = [#map_bcast_d0, #map_identity],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%b_inner : tensor<1x8x4xf32>) outs(%empty_inner_bcast_b : tensor<8x8x4xf32>) {
    ^bb0(%in: f32, %out_elem: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x4xf32>

    // Element-wise multiply
    %empty_inner_mul = tensor.empty() : tensor<8x8x4xf32>
    %inner_mul = linalg.generic {
      indexing_maps = [#map_identity, #map_identity, #map_identity],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%inner_bcast_a, %inner_bcast_b : tensor<8x8x4xf32>, tensor<8x8x4xf32>)
      outs(%empty_inner_mul : tensor<8x8x4xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
      %prod = arith.mulf %in0, %in1 : f32
      linalg.yield %prod : f32
    } -> tensor<8x8x4xf32>

    // Inner reduce (dim=2, additive) — ROOT inside scf.for body
    // Uses %acc (iter_arg) as init, connecting to the outer scope through
    // iter_args. Despite this connection, it is in a DIFFERENT block and
    // must be treated as an independent root (Req 1.7).
    %inner_reduced = linalg.reduce ins(%inner_mul : tensor<8x8x4xf32>)
                                   outs(%acc : tensor<8x8xf32>)
                                   dimensions = [2]
      (%in: f32, %iacc: f32) {
        %sum = arith.addf %in, %iacc : f32
        linalg.yield %sum : f32
      }

    scf.yield %inner_reduced : tensor<8x8xf32>
  }

  // Combine results (element-wise add of outer_reduced and loop_result)
  %empty_final = tensor.empty() : tensor<8x8xf32>
  %final = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%outer_reduced, %loop_result : tensor<8x8xf32>, tensor<8x8xf32>)
    outs(%empty_final : tensor<8x8xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  } -> tensor<8x8xf32>

  return %final : tensor<8x8xf32>
}

// Verify that BOTH reduces are treated as independent roots and tiled.
// Each contraction pattern has output shape 8x8 with tile-m=4, tile-n=4,
// so both M and N get tiled (8 > 4).
//
// CHECK-LABEL: func.func @chain_scope_block
//
// The outer reduce chain is tiled: nested scf.for loops for M and N
// (from 0 to 8 step 4 for both dimensions).
// The tiling pass creates its own constants (e.g. %c0_0, %c8, %c4),
// so we match the literal constant values in the loop bounds.
// CHECK: scf.for %{{.*}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK:   scf.for %{{.*}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.reduce
// CHECK:       arith.addf
//
// The original iteration loop (scf.for with step 1) still exists
// CHECK: scf.for %{{.*}} = %c0{{[_0-9]*}} to %c3{{[_0-9]*}} step %c1{{[_0-9]*}}
//
// Inside the iteration loop, the inner reduce chain is also tiled:
// nested scf.for loops for M and N (from 0 to 8 step 4).
// CHECK:   scf.for %{{.*}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK:     scf.for %{{.*}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK:       linalg.generic
// CHECK:       linalg.generic
// CHECK:       linalg.generic
// CHECK:       linalg.reduce
// CHECK:         arith.addf
