// RUN: buddy-opt %s > %t.no-pass
// RUN: buddy-opt --qgemm-recognize %s > %t.with-pass
// RUN: diff -u %t.no-pass %t.with-pass
// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// This is the three-level reduction chain from synthetic_l3_canonical. Every
// level exposes its quantization block with expand_shape before reducing; the
// final plain reduction must be the only candidate root.

#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK-LABEL: func.func @three_level_chain
// CHECK: tensor.expand_shape
// CHECK: linalg.reduce
// CHECK: tensor.expand_shape
// CHECK: linalg.reduce
// CHECK: tensor.expand_shape
// CHECK: linalg.reduce
// CHECK: linalg.reduce
// CHECK-NOT: qgemm.node
func.func @three_level_chain(%lhs: tensor<32x1x1024xf32>,
                              %rhs: tensor<1x16x1024xf32>,
                              %factor1: tensor<32x16x32xf32>,
                              %factor2: tensor<32x16x8xf32>,
                              %factor3: tensor<32x16x2xf32>) -> tensor<32x16xf32> {
  %zero = arith.constant 0.0 : f32
  %product_empty = tensor.empty() : tensor<32x16x1024xf32>
  %product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<32x1x1024xf32>, tensor<1x16x1024xf32>)
      outs(%product_empty : tensor<32x16x1024xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<32x16x1024xf32>

  %expanded1 = tensor.expand_shape %product [[0], [1], [2, 3]]
      output_shape [32, 16, 32, 32]
      : tensor<32x16x1024xf32> into tensor<32x16x32x32xf32>
  %reduce1_empty = tensor.empty() : tensor<32x16x32xf32>
  %reduce1_init = linalg.fill ins(%zero : f32)
      outs(%reduce1_empty : tensor<32x16x32xf32>) -> tensor<32x16x32xf32>
  %reduce1 = linalg.reduce ins(%expanded1 : tensor<32x16x32x32xf32>)
      outs(%reduce1_init : tensor<32x16x32xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %scaled1_empty = tensor.empty() : tensor<32x16x32xf32>
  %scaled1 = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reduce1, %factor1 : tensor<32x16x32xf32>, tensor<32x16x32xf32>)
      outs(%scaled1_empty : tensor<32x16x32xf32>) {
  ^bb0(%value: f32, %factor: f32, %out: f32):
    %scaled = arith.mulf %value, %factor : f32
    linalg.yield %scaled : f32
  } -> tensor<32x16x32xf32>

  %expanded2 = tensor.expand_shape %scaled1 [[0], [1], [2, 3]]
      output_shape [32, 16, 8, 4]
      : tensor<32x16x32xf32> into tensor<32x16x8x4xf32>
  %reduce2_empty = tensor.empty() : tensor<32x16x8xf32>
  %reduce2_init = linalg.fill ins(%zero : f32)
      outs(%reduce2_empty : tensor<32x16x8xf32>) -> tensor<32x16x8xf32>
  %reduce2 = linalg.reduce ins(%expanded2 : tensor<32x16x8x4xf32>)
      outs(%reduce2_init : tensor<32x16x8xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %scaled2_empty = tensor.empty() : tensor<32x16x8xf32>
  %scaled2 = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reduce2, %factor2 : tensor<32x16x8xf32>, tensor<32x16x8xf32>)
      outs(%scaled2_empty : tensor<32x16x8xf32>) {
  ^bb0(%value: f32, %factor: f32, %out: f32):
    %scaled = arith.mulf %value, %factor : f32
    linalg.yield %scaled : f32
  } -> tensor<32x16x8xf32>

  %expanded3 = tensor.expand_shape %scaled2 [[0], [1], [2, 3]]
      output_shape [32, 16, 2, 4]
      : tensor<32x16x8xf32> into tensor<32x16x2x4xf32>
  %reduce3_empty = tensor.empty() : tensor<32x16x2xf32>
  %reduce3_init = linalg.fill ins(%zero : f32)
      outs(%reduce3_empty : tensor<32x16x2xf32>) -> tensor<32x16x2xf32>
  %reduce3 = linalg.reduce ins(%expanded3 : tensor<32x16x2x4xf32>)
      outs(%reduce3_init : tensor<32x16x2xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %scaled3_empty = tensor.empty() : tensor<32x16x2xf32>
  %scaled3 = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reduce3, %factor3 : tensor<32x16x2xf32>, tensor<32x16x2xf32>)
      outs(%scaled3_empty : tensor<32x16x2xf32>) {
  ^bb0(%value: f32, %factor: f32, %out: f32):
    %scaled = arith.mulf %value, %factor : f32
    linalg.yield %scaled : f32
  } -> tensor<32x16x2xf32>

  %final_empty = tensor.empty() : tensor<32x16xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<32x16xf32>) -> tensor<32x16xf32>
  %final = linalg.reduce ins(%scaled3 : tensor<32x16x2xf32>)
      outs(%final_init : tensor<32x16xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<32x16xf32>
}
