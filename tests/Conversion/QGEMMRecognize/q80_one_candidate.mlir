// RUN: buddy-opt %s > %t.no-pass
// RUN: buddy-opt --qgemm-recognize %s > %t.with-pass
// RUN: diff -u %t.no-pass %t.with-pass
// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// A small Q8_0-shaped canonical fragment. The two two-dimensional inputs are
// expanded to three dimensions before independent broadcasts. The product is
// split into one quantization block, reduced, scaled, and finally reduced.

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] op 0 walk: ops=9 boundaries=9
// CHECK-LABEL: func.func @q80_one_candidate
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: tensor.expand_shape
// CHECK: linalg.reduce
// CHECK: arith.mulf
// CHECK: linalg.reduce
// CHECK-NOT: qgemm.node
func.func @q80_one_candidate(%arg0: tensor<8x32xf32>,
                              %arg1: tensor<8x32xf32>,
                              %factor: tensor<8x8x1xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32

  %arg0_3d = tensor.expand_shape %arg0 [[0], [1, 2]]
      output_shape [8, 1, 32]
      : tensor<8x32xf32> into tensor<8x1x32xf32>
  %arg1_3d = tensor.expand_shape %arg1 [[0, 1], [2]]
      output_shape [1, 8, 32]
      : tensor<8x32xf32> into tensor<1x8x32xf32>

  %broadcast0_empty = tensor.empty() : tensor<8x8x32xf32>
  %broadcast0 = linalg.generic {
    indexing_maps = [#broadcast_m, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0_3d : tensor<8x1x32xf32>)
      outs(%broadcast0_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %broadcast1_empty = tensor.empty() : tensor<8x8x32xf32>
  %broadcast1 = linalg.generic {
    indexing_maps = [#broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg1_3d : tensor<1x8x32xf32>)
      outs(%broadcast1_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%broadcast0, %broadcast1 : tensor<8x8x32xf32>, tensor<8x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<8x8x32xf32>

  %split = tensor.expand_shape %product [[0], [1], [2, 3]]
      output_shape [8, 8, 1, 32]
      : tensor<8x8x32xf32> into tensor<8x8x1x32xf32>
  %inner_empty = tensor.empty() : tensor<8x8x1xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<8x8x1xf32>) -> tensor<8x8x1xf32>
  %inner = linalg.reduce ins(%split : tensor<8x8x1x32xf32>)
      outs(%inner_init : tensor<8x8x1xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %scaled_empty = tensor.empty() : tensor<8x8x1xf32>
  %scaled = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner, %factor : tensor<8x8x1xf32>, tensor<8x8x1xf32>)
      outs(%scaled_empty : tensor<8x8x1xf32>) {
  ^bb0(%in: f32, %scale: f32, %out: f32):
    %value = arith.mulf %in, %scale : f32
    linalg.yield %value : f32
  } -> tensor<8x8x1xf32>

  %final_empty = tensor.empty() : tensor<8x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %outer = linalg.reduce ins(%scaled : tensor<8x8x1xf32>)
      outs(%final_init : tensor<8x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %outer : tensor<8x8xf32>
}
