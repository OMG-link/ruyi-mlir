// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// The candidate reduction is inside an scf.for body. Function arguments and
// the loop iter_arg are block/cross-block boundaries; the walk must not escape
// the loop body to the outer fill or scf.for operation.

#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#identity2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] op 0 walk: ops=4 boundaries=6
// CHECK-LABEL: func.func @walk_block_boundary
// CHECK: scf.for
// CHECK: linalg.reduce
// CHECK-SAME: qgemm.node = {root = 0 : i64, stage = "compute"}
// CHECK-SAME: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
func.func @walk_block_boundary(%a: tensor<8x1x4xf32>,
                                %b: tensor<1x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %outer_empty = tensor.empty() : tensor<8x8xf32>
  %outer_init = linalg.fill ins(%zero : f32)
      outs(%outer_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %result = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %outer_init) -> (tensor<8x8xf32>) {
    %a_empty = tensor.empty() : tensor<8x8x4xf32>
    %a_broadcast = linalg.generic {
      indexing_maps = [#broadcast_m, #identity3],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a : tensor<8x1x4xf32>)
        outs(%a_empty : tensor<8x8x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x4xf32>
    %b_empty = tensor.empty() : tensor<8x8x4xf32>
    %b_broadcast = linalg.generic {
      indexing_maps = [#broadcast_n, #identity3],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%b : tensor<1x8x4xf32>)
        outs(%b_empty : tensor<8x8x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x4xf32>
    %product_empty = tensor.empty() : tensor<8x8x4xf32>
    %product = linalg.generic {
      indexing_maps = [#identity3, #identity3, #identity3],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a_broadcast, %b_broadcast : tensor<8x8x4xf32>, tensor<8x8x4xf32>)
        outs(%product_empty : tensor<8x8x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %value = arith.mulf %lhs, %rhs : f32
      linalg.yield %value : f32
    } -> tensor<8x8x4xf32>
    %reduced = linalg.reduce ins(%product : tensor<8x8x4xf32>)
        outs(%acc : tensor<8x8xf32>) dimensions = [2]
      (%in: f32, %accumulator: f32) {
        %sum = arith.addf %in, %accumulator : f32
        linalg.yield %sum : f32
      }
    scf.yield %reduced : tensor<8x8xf32>
  }
  return %result : tensor<8x8xf32>
}
