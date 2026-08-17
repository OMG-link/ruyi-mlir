// RUN: buddy-opt %s > %t.no-pass
// RUN: buddy-opt --qgemm-recognize %s > %t.with-pass
// RUN: diff -u %t.no-pass %t.with-pass
// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// This is the two-path level-1 fragment of q4k_q8k_canonical. Both paths
// expose a block with expand_shape and reduce, then meet at the subtraction
// merge before the final reduction.

#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK-LABEL: func.func @two_path_merge
// CHECK: tensor.expand_shape
// CHECK: linalg.reduce
// CHECK: tensor.expand_shape
// CHECK: linalg.reduce
// CHECK: arith.subf
// CHECK: linalg.reduce
// CHECK-NOT: qgemm.node
func.func @two_path_merge(%main_lhs: tensor<192x1x512xf32>,
                           %main_rhs: tensor<1x32x512xf32>,
                           %zero_lhs: tensor<192x1x512xf32>,
                           %zero_rhs: tensor<1x32x512xf32>) -> tensor<192x32xf32> {
  %zero = arith.constant 0.0 : f32

  %main_product_empty = tensor.empty() : tensor<192x32x512xf32>
  %main_product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%main_lhs, %main_rhs : tensor<192x1x512xf32>, tensor<1x32x512xf32>)
      outs(%main_product_empty : tensor<192x32x512xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<192x32x512xf32>
  %main_expanded = tensor.expand_shape %main_product [[0], [1], [2, 3]]
      output_shape [192, 32, 16, 32]
      : tensor<192x32x512xf32> into tensor<192x32x16x32xf32>
  %main_empty = tensor.empty() : tensor<192x32x16xf32>
  %main_init = linalg.fill ins(%zero : f32)
      outs(%main_empty : tensor<192x32x16xf32>) -> tensor<192x32x16xf32>
  %main_reduce = linalg.reduce
      ins(%main_expanded : tensor<192x32x16x32xf32>)
      outs(%main_init : tensor<192x32x16xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %zero_product_empty = tensor.empty() : tensor<192x32x512xf32>
  %zero_product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%zero_lhs, %zero_rhs : tensor<192x1x512xf32>, tensor<1x32x512xf32>)
      outs(%zero_product_empty : tensor<192x32x512xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<192x32x512xf32>
  %zero_expanded = tensor.expand_shape %zero_product [[0], [1], [2, 3]]
      output_shape [192, 32, 16, 32]
      : tensor<192x32x512xf32> into tensor<192x32x16x32xf32>
  %zero_empty = tensor.empty() : tensor<192x32x16xf32>
  %zero_init = linalg.fill ins(%zero : f32)
      outs(%zero_empty : tensor<192x32x16xf32>) -> tensor<192x32x16xf32>
  %zero_reduce = linalg.reduce
      ins(%zero_expanded : tensor<192x32x16x32xf32>)
      outs(%zero_init : tensor<192x32x16xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %merged_empty = tensor.empty() : tensor<192x32x16xf32>
  %merged = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%main_reduce, %zero_reduce : tensor<192x32x16xf32>, tensor<192x32x16xf32>)
      outs(%merged_empty : tensor<192x32x16xf32>) {
  ^bb0(%main: f32, %zero_path: f32, %out: f32):
    %difference = arith.subf %main, %zero_path : f32
    linalg.yield %difference : f32
  } -> tensor<192x32x16xf32>

  %final_empty = tensor.empty() : tensor<192x32xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<192x32xf32>) -> tensor<192x32xf32>
  %final = linalg.reduce ins(%merged : tensor<192x32x16xf32>)
      outs(%final_init : tensor<192x32xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<192x32xf32>
}
