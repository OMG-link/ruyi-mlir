// RUN: timeout 8 buddy-opt %s > %t.before
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %s > %t.after
// RUN: awk '/func.func @outside/{capture=1} capture {print} capture && /^  }/{exit}' %t.before > %t.before.outside
// RUN: awk '/func.func @outside/{capture=1} capture {print} capture && /^  }/{exit}' %t.after > %t.after.outside
// RUN: diff -u %t.before.outside %t.after.outside
// RUN: FileCheck %s < %t.after

// The target function is in the coarse QGEMM scope; the unrelated function is
// outside it and must print byte-for-byte identically before and after the pass.
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @scope_target(%lhs: tensor<2x2x4xf32>,
                          %rhs: tensor<2x2x4xf32>) -> tensor<2x2xf32> {
    %zero = arith.constant 0.0 : f32
    %product_empty = tensor.empty() : tensor<2x2x4xf32>
    %product = linalg.generic {
      indexing_maps = [#identity3, #identity3, #identity3],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<2x2x4xf32>, tensor<2x2x4xf32>)
        outs(%product_empty : tensor<2x2x4xf32>) {
    ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
      %value = arith.mulf %lhs_value, %rhs_value : f32
      linalg.yield %value : f32
    } -> tensor<2x2x4xf32>

    %expanded = tensor.expand_shape %product [[0], [1], [2, 3]]
        output_shape [2, 2, 2, 2]
        : tensor<2x2x4xf32> into tensor<2x2x2x2xf32>
    %inner_empty = tensor.empty() : tensor<2x2x2xf32>
    %inner_init = linalg.fill ins(%zero : f32)
        outs(%inner_empty : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    %inner = linalg.reduce ins(%expanded : tensor<2x2x2x2xf32>)
        outs(%inner_init : tensor<2x2x2xf32>) dimensions = [3]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }

    %final_empty = tensor.empty() : tensor<2x2xf32>
    %final_init = linalg.fill ins(%zero : f32)
        outs(%final_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    %final = linalg.reduce ins(%inner : tensor<2x2x2xf32>)
        outs(%final_init : tensor<2x2xf32>) dimensions = [2]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
    return %final : tensor<2x2xf32>
  }

  func.func @outside(%arg: tensor<4xf32>) -> tensor<4xf32> {
    return %arg : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @scope_target
// CHECK: linalg.generic {{.*}}tensor<2x2x2x2xf32>
// CHECK: arith.mulf
// CHECK-LABEL: func.func @outside
// CHECK-NEXT: return %{{.*}} : tensor<4xf32>
