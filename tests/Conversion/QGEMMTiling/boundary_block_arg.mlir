// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s | FileCheck %s
//
// NOTE: This test requires the v02 pass restructuring (task 6.1) which replaces
// FinalConsumerFinder with ReduceChainAnalysis. Until that restructuring is
// complete, this test will not pass.
//
// Tests for BFS boundary condition: block arguments (Requirement 4.2)
//
// Verifies that when the backward BFS encounters a block argument (including
// scf.for iter_args), it treats it as a leaf node and stops traversal at
// that point. The BFS should NOT try to traverse outside the loop body.
//
// Test scenario: An scf.for loop where the contraction pattern inside the
// loop body uses a block argument (iter_arg) as the init for the reduce,
// and function arguments passed into the loop body as inputs to the
// broadcasts. The BFS should stop at block arguments and collect only the
// contraction pattern within the loop body.
//
// The contraction pattern uses 3D tensors with broadcast on dim 0 and dim 1,
// giving exactly 2 broadcast dims. Output shape is 8x8, with tile-m=4 tile-n=4:
//   M=8 > 4 → tiled with step 4
//   N=8 > 4 → tiled with step 4

#map_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast_d0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map_bcast_d1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

// CHECK-LABEL: func.func @boundary_block_arg
func.func @boundary_block_arg(%a: tensor<8x1x4xf32>,
                               %b: tensor<1x8x4xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // Initial accumulator for the loop
  %empty_init = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty_init : tensor<8x8xf32>) -> tensor<8x8xf32>

  // scf.for loop: the contraction pattern inside uses block arguments.
  // The BFS should stop at block arguments (%arg_a, %arg_b are function args
  // passed through; %acc is an iter_arg) and not traverse outside the loop.
  %result = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (tensor<8x8xf32>) {

    // Broadcast %a on dim 1: (8,1,4) → (8,8,4)
    // %a is a function argument — BFS stops here (block arg boundary from
    // the perspective of the loop body block, since %a is defined outside).
    %empty_bcast_a = tensor.empty() : tensor<8x8x4xf32>
    %bcast_a = linalg.generic {
      indexing_maps = [#map_bcast_d1, #map_identity],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a : tensor<8x1x4xf32>) outs(%empty_bcast_a : tensor<8x8x4xf32>) {
    ^bb0(%in: f32, %out_elem: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x4xf32>

    // Broadcast %b on dim 0: (1,8,4) → (8,8,4)
    // %b is a function argument — BFS stops here (same reason).
    %empty_bcast_b = tensor.empty() : tensor<8x8x4xf32>
    %bcast_b = linalg.generic {
      indexing_maps = [#map_bcast_d0, #map_identity],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%b : tensor<1x8x4xf32>) outs(%empty_bcast_b : tensor<8x8x4xf32>) {
    ^bb0(%in: f32, %out_elem: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x4xf32>

    // Element-wise multiply
    %empty_mul = tensor.empty() : tensor<8x8x4xf32>
    %mul = linalg.generic {
      indexing_maps = [#map_identity, #map_identity, #map_identity],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%bcast_a, %bcast_b : tensor<8x8x4xf32>, tensor<8x8x4xf32>)
      outs(%empty_mul : tensor<8x8x4xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
      %prod = arith.mulf %in0, %in1 : f32
      linalg.yield %prod : f32
    } -> tensor<8x8x4xf32>

    // Additive reduce on dim 2 — tiling root inside the loop body.
    // Uses %acc (iter_arg / block argument) as init.
    // BFS should stop at %acc since it is a block argument.
    %reduced = linalg.reduce ins(%mul : tensor<8x8x4xf32>)
                             outs(%acc : tensor<8x8xf32>)
                             dimensions = [2]
      (%in: f32, %iacc: f32) {
        %sum = arith.addf %in, %iacc : f32
        linalg.yield %sum : f32
      }

    scf.yield %reduced : tensor<8x8xf32>
  }

  return %result : tensor<8x8xf32>
}

// Verify that tiling is applied to the contraction pattern inside the loop body.
// The reduce has output shape 8x8, tile-m=4 tile-n=4 → both M and N tiled.
//
// The original iteration loop (scf.for with step 1, iterating 4 times) exists.
// CHECK: scf.for %{{.*}} = %c0{{[_0-9]*}} to %c4{{[_0-9]*}} step %c1{{[_0-9]*}}
//
// Inside the iteration loop, the contraction pattern is tiled:
// nested scf.for loops for M and N (from 0 to 8 step 4).
// The tiling pass creates its own constants (e.g. %c0_0, %c8, %c4_3),
// so we match the constant names with optional suffixes.
// CHECK:   scf.for %{{.*}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK:     scf.for %{{.*}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK:       linalg.generic
// CHECK:       linalg.generic
// CHECK:       linalg.generic
// CHECK:       linalg.reduce
// CHECK:         arith.addf
