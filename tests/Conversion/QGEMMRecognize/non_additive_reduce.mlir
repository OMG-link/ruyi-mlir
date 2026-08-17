// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// Non-additive combiners are filtered during candidate enumeration. They do
// not produce a rejection record or a diagnostic, even with the default
// explain=false setting.

// CHECK: [qgemm-recognize] candidates: 0
// CHECK-LABEL: func.func @non_additive_reduce
// CHECK: linalg.reduce
// CHECK: arith.maxnumf
// CHECK-NOT: qgemm.node
// CHECK-NOT: remark
func.func @non_additive_reduce(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<2xf32>
  %init = linalg.fill ins(%neg_inf : f32) outs(%empty : tensor<2xf32>)
      -> tensor<2xf32>
  %result = linalg.reduce ins(%arg0 : tensor<2x4xf32>)
      outs(%init : tensor<2xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
      %max = arith.maxnumf %in, %acc : f32
      linalg.yield %max : f32
    }
  return %result : tensor<2xf32>
}
