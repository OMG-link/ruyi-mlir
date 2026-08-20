// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %s | FileCheck %s

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#identity5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#broadcast_m4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#broadcast_n4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

// The adjacent expansion and collapse have a direct reassociation from the
// source to the result, so the chain is reduced to one expansion.
// CHECK-LABEL: func.func @reshape_chain_one_op
// CHECK: tensor.expand_shape %{{.*}} {{\[\[0\], \[1, 2\]\]}} output_shape [192, 16, 32] : tensor<192x512xf32> into tensor<192x16x32xf32>
// CHECK-NOT: tensor.collapse_shape
func.func @reshape_chain_one_op(%activation: tensor<192x512xf32>,
                                 %weight: tensor<8x16x32xf32>)
    -> tensor<192x8xf32> {
  %zero = arith.constant 0.0 : f32
  %expanded = tensor.expand_shape %activation [[0], [1, 2, 3]]
      output_shape [192, 1, 16, 32]
      : tensor<192x512xf32> into tensor<192x1x16x32xf32>
  %reshaped = tensor.collapse_shape %expanded [[0], [1, 2], [3]]
      : tensor<192x1x16x32xf32> into tensor<192x16x32xf32>

  %lhs_empty = tensor.empty() : tensor<192x8x16x32xf32>
  %lhs = linalg.generic {
    indexing_maps = [#broadcast_m4, #identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%reshaped : tensor<192x16x32xf32>)
      outs(%lhs_empty : tensor<192x8x16x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<192x8x16x32xf32>
  %rhs_empty = tensor.empty() : tensor<192x8x16x32xf32>
  %rhs = linalg.generic {
    indexing_maps = [#broadcast_n4, #identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%weight : tensor<8x16x32xf32>)
      outs(%rhs_empty : tensor<192x8x16x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<192x8x16x32xf32>
  %product_empty = tensor.empty() : tensor<192x8x16x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity4, #identity4, #identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<192x8x16x32xf32>, tensor<192x8x16x32xf32>)
      outs(%product_empty : tensor<192x8x16x32xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<192x8x16x32xf32>
  %inner_empty = tensor.empty() : tensor<192x8x16xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<192x8x16xf32>) -> tensor<192x8x16xf32>
  %inner = linalg.reduce ins(%product : tensor<192x8x16x32xf32>)
      outs(%inner_init : tensor<192x8x16xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %final_empty = tensor.empty() : tensor<192x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<192x8xf32>) -> tensor<192x8xf32>
  %final = linalg.reduce ins(%inner : tensor<192x8x16xf32>)
      outs(%final_init : tensor<192x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<192x8xf32>
}

// The ends have equal rank and cannot be connected by one reassociative
// reshape. A three-member chain therefore uses the flat two-operation form.
// CHECK-LABEL: func.func @reshape_chain_two_op
// CHECK: %[[FLAT:[A-Za-z0-9_]+]] = tensor.collapse_shape %{{.*}} {{\[\[0, 1, 2, 3\]\]}} : tensor<2x2x4x32xf32> into tensor<512xf32>
// CHECK-NEXT: %{{.*}} = tensor.expand_shape %[[FLAT]] {{\[\[0, 1, 2, 3\]\]}} output_shape [2, 8, 1, 32] : tensor<512xf32> into tensor<2x8x1x32xf32>
func.func @reshape_chain_two_op(%source: tensor<2x2x4x32xf32>,
                                 %other: tensor<2x8x1x32xf32>)
    -> tensor<2x8xf32> {
  %zero = arith.constant 0.0 : f32
  %expanded = tensor.expand_shape %source [[0], [1], [2], [3, 4]]
      output_shape [2, 2, 4, 1, 32]
      : tensor<2x2x4x32xf32> into tensor<2x2x4x1x32xf32>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3], [4]]
      : tensor<2x2x4x1x32xf32> into tensor<2x8x32xf32>
  %reshaped = tensor.expand_shape %collapsed [[0], [1], [2, 3]]
      output_shape [2, 8, 1, 32]
      : tensor<2x8x32xf32> into tensor<2x8x1x32xf32>

  %product_empty = tensor.empty() : tensor<2x8x1x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity4, #identity4, #identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%reshaped, %other : tensor<2x8x1x32xf32>, tensor<2x8x1x32xf32>)
      outs(%product_empty : tensor<2x8x1x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<2x8x1x32xf32>
  %inner_empty = tensor.empty() : tensor<2x8x1xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<2x8x1xf32>) -> tensor<2x8x1xf32>
  %inner = linalg.reduce ins(%product : tensor<2x8x1x32xf32>)
      outs(%inner_init : tensor<2x8x1xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %final_empty = tensor.empty() : tensor<2x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<2x8xf32>) -> tensor<2x8xf32>
  %final = linalg.reduce ins(%inner : tensor<2x8x1xf32>)
      outs(%final_init : tensor<2x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<2x8xf32>
}

// With only two members and equal-rank ends, the chain is already at the
// global minimum and must remain unchanged.
// CHECK-LABEL: func.func @reshape_chain_minimal
// CHECK: %[[MIN_EXPAND:[A-Za-z0-9_]+]] = tensor.expand_shape %{{.*}} {{\[\[0\], \[1\], \[2\], \[3, 4\]\]}} output_shape [2, 2, 4, 1, 32] : tensor<2x2x4x32xf32> into tensor<2x2x4x1x32xf32>
// CHECK-NEXT: %{{.*}} = tensor.collapse_shape %[[MIN_EXPAND]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} : tensor<2x2x4x1x32xf32> into tensor<2x8x1x32xf32>
// CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<2x8x1xf32>
func.func @reshape_chain_minimal(%source: tensor<2x2x4x32xf32>)
    -> tensor<2x8xf32> {
  %zero = arith.constant 0.0 : f32
  %expanded = tensor.expand_shape %source [[0], [1], [2], [3, 4]]
      output_shape [2, 2, 4, 1, 32]
      : tensor<2x2x4x32xf32> into tensor<2x2x4x1x32xf32>
  %reshaped = tensor.collapse_shape %expanded [[0], [1, 2], [3], [4]]
      : tensor<2x2x4x1x32xf32> into tensor<2x8x1x32xf32>
  %insert_empty = tensor.empty() : tensor<2x8x1x32xf32>
  %inserted = tensor.insert_slice %reshaped into %insert_empty[0, 0, 0, 0]
      [2, 8, 1, 32] [1, 1, 1, 1]
      : tensor<2x8x1x32xf32> into tensor<2x8x1x32xf32>
  %inner_empty = tensor.empty() : tensor<2x8x1xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<2x8x1xf32>) -> tensor<2x8x1xf32>
  %inner = linalg.reduce ins(%inserted : tensor<2x8x1x32xf32>)
      outs(%inner_init : tensor<2x8x1xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %final_empty = tensor.empty() : tensor<2x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<2x8xf32>) -> tensor<2x8xf32>
  %final = linalg.reduce ins(%inner : tensor<2x8x1xf32>)
      outs(%final_init : tensor<2x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<2x8xf32>
}

// The middle value has a second consumer, so the original two reshape
// operations cannot be replaced by the one-operation form.
// CHECK-LABEL: func.func @reshape_chain_multi_use
// CHECK: %[[SHARED:[A-Za-z0-9_]+]] = tensor.expand_shape %{{.*}} {{\[\[0\], \[1\], \[2\], \[3, 4\]\]}} output_shape [2, 2, 4, 1, 32] : tensor<2x2x4x32xf32> into tensor<2x2x4x1x32xf32>
// CHECK: tensor.extract %[[SHARED]]
// CHECK: tensor.collapse_shape %[[SHARED]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} : tensor<2x2x4x1x32xf32> into tensor<2x8x1x32xf32>
func.func @reshape_chain_multi_use(%source: tensor<2x2x4x32xf32>)
    -> (tensor<2x8xf32>, f32) {
  %zero = arith.constant 0.0 : f32
  %index = arith.constant 0 : index
  %expanded = tensor.expand_shape %source [[0], [1], [2], [3, 4]]
      output_shape [2, 2, 4, 1, 32]
      : tensor<2x2x4x32xf32> into tensor<2x2x4x1x32xf32>
  %side = tensor.extract %expanded[%index, %index, %index, %index, %index]
      : tensor<2x2x4x1x32xf32>
  %reshaped = tensor.collapse_shape %expanded [[0], [1, 2], [3], [4]]
      : tensor<2x2x4x1x32xf32> into tensor<2x8x1x32xf32>
  %insert_empty = tensor.empty() : tensor<2x8x1x32xf32>
  %inserted = tensor.insert_slice %reshaped into %insert_empty[0, 0, 0, 0]
      [2, 8, 1, 32] [1, 1, 1, 1]
      : tensor<2x8x1x32xf32> into tensor<2x8x1x32xf32>
  %inner_empty = tensor.empty() : tensor<2x8x1xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<2x8x1xf32>) -> tensor<2x8x1xf32>
  %inner = linalg.reduce ins(%inserted : tensor<2x8x1x32xf32>)
      outs(%inner_init : tensor<2x8x1xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %final_empty = tensor.empty() : tensor<2x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<2x8xf32>) -> tensor<2x8xf32>
  %final = linalg.reduce ins(%inner : tensor<2x8x1xf32>)
      outs(%final_init : tensor<2x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final, %side : tensor<2x8xf32>, f32
}
