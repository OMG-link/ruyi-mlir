// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// A Q4_0-style decoder fragment cut from the captured shape chain. The
// walker must admit insert_slice, transpose, reshape, collapse_shape and
// expand_shape before reaching the product and reduction.

#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] op 0 walk: ops=11 boundaries=10
// CHECK-LABEL: func.func @q40_walk
// CHECK: tensor.insert_slice
// CHECK: linalg.transpose
// CHECK: tensor.reshape
// CHECK: tensor.collapse_shape
// CHECK: tensor.expand_shape
// CHECK: qgemm.node = {root = 0 : i64, stage = "compute"}
func.func @q40_walk(%activation: tensor<8x1x32xf32>,
                     %packed0: tensor<1x8x16xf32>,
                     %packed1: tensor<1x8x16xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %packed_empty = tensor.empty() : tensor<1x8x16x2xf32>
  %inserted0 = tensor.insert_slice %packed0 into %packed_empty[0, 0, 0, 0]
      [1, 8, 16, 1] [1, 1, 1, 1]
      : tensor<1x8x16xf32> into tensor<1x8x16x2xf32>
  %inserted1 = tensor.insert_slice %packed1 into %inserted0[0, 0, 0, 1]
      [1, 8, 16, 1] [1, 1, 1, 1]
      : tensor<1x8x16xf32> into tensor<1x8x16x2xf32>
  %transposed_empty = tensor.empty() : tensor<1x2x8x16xf32>
  %transposed = linalg.transpose ins(%inserted1 : tensor<1x8x16x2xf32>)
      outs(%transposed_empty : tensor<1x2x8x16xf32>) permutation = [0, 3, 1, 2]
  %shape = arith.constant dense<[1, 8, 32]> : tensor<3xi64>
  %reshaped = tensor.reshape %transposed(%shape)
      : (tensor<1x2x8x16xf32>, tensor<3xi64>) -> tensor<1x8x32xf32>
  %collapsed = tensor.collapse_shape %reshaped [[0, 1], [2]]
      : tensor<1x8x32xf32> into tensor<8x32xf32>
  %decoded = tensor.expand_shape %collapsed [[0, 1], [2]]
      output_shape [1, 8, 32]
      : tensor<8x32xf32> into tensor<1x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%activation, %decoded : tensor<8x1x32xf32>, tensor<1x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<8x8x32xf32>
  %expanded = tensor.expand_shape %product [[0], [1], [2, 3]]
      output_shape [8, 8, 4, 8]
      : tensor<8x8x32xf32> into tensor<8x8x4x8xf32>
  %inner_empty = tensor.empty() : tensor<8x8x4xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<8x8x4xf32>) -> tensor<8x8x4xf32>
  %inner = linalg.reduce ins(%expanded : tensor<8x8x4x8xf32>)
      outs(%inner_init : tensor<8x8x4xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %scaled_empty = tensor.empty() : tensor<8x8x4xf32>
  %scaled = linalg.generic {
    indexing_maps = [#identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner : tensor<8x8x4xf32>)
      outs(%scaled_empty : tensor<8x8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>
  %final_empty = tensor.empty() : tensor<8x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %final = linalg.reduce ins(%scaled : tensor<8x8x4xf32>)
      outs(%final_init : tensor<8x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<8x8xf32>
}
