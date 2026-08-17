// RUN: buddy-opt %s > %t.no-pass
// RUN: buddy-opt --qgemm-recognize %s > %t.with-pass
// RUN: diff -u %t.no-pass %t.with-pass
// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// A compact Q4_K-style diamond. Both paths share the activation broadcast and
// the linalg.fill destination used to initialize their reductions. The walk
// must visit shared operations once, follow both merge operands, and stop at
// the three tensor block arguments.

#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] op 0 walk: ops=11 boundaries=9
// CHECK-LABEL: func.func @q4k_shared_walk
// CHECK: arith.subf
// CHECK: linalg.reduce
// CHECK-NOT: qgemm.node
func.func @q4k_shared_walk(%activation: tensor<192x1x512xf32>,
                            %main_weight: tensor<1x32x512xf32>,
                            %zero_weight: tensor<1x32x512xf32>)
    -> tensor<192x32xf32> {
  %zero = arith.constant 0.0 : f32

  %activation_empty = tensor.empty() : tensor<192x32x512xf32>
  %activation_broadcast = linalg.generic {
    indexing_maps = [#broadcast_m, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%activation : tensor<192x1x512xf32>)
      outs(%activation_empty : tensor<192x32x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<192x32x512xf32>

  %main_product_empty = tensor.empty() : tensor<192x32x512xf32>
  %main_product = linalg.generic {
    indexing_maps = [#identity3, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%activation_broadcast, %main_weight : tensor<192x32x512xf32>, tensor<1x32x512xf32>)
      outs(%main_product_empty : tensor<192x32x512xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<192x32x512xf32>
  %main_expanded = tensor.expand_shape %main_product [[0], [1], [2, 3]]
      output_shape [192, 32, 16, 32]
      : tensor<192x32x512xf32> into tensor<192x32x16x32xf32>

  %shared_empty = tensor.empty() : tensor<192x32x16xf32>
  %shared_init = linalg.fill ins(%zero : f32)
      outs(%shared_empty : tensor<192x32x16xf32>) -> tensor<192x32x16xf32>
  %main_reduce = linalg.reduce
      ins(%main_expanded : tensor<192x32x16x32xf32>)
      outs(%shared_init : tensor<192x32x16xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %zero_product_empty = tensor.empty() : tensor<192x32x512xf32>
  %zero_product = linalg.generic {
    indexing_maps = [#identity3, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%activation_broadcast, %zero_weight : tensor<192x32x512xf32>, tensor<1x32x512xf32>)
      outs(%zero_product_empty : tensor<192x32x512xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<192x32x512xf32>
  %zero_expanded = tensor.expand_shape %zero_product [[0], [1], [2, 3]]
      output_shape [192, 32, 16, 32]
      : tensor<192x32x512xf32> into tensor<192x32x16x32xf32>
  %zero_reduce = linalg.reduce
      ins(%zero_expanded : tensor<192x32x16x32xf32>)
      outs(%shared_init : tensor<192x32x16xf32>) dimensions = [3]
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
