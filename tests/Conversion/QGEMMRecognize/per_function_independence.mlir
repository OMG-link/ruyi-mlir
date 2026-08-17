// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// Candidate numbering and counts restart per function. The second function
// has no reduction and must not affect the first function's one candidate.

// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] candidates: 0
// CHECK-LABEL: func.func @one_candidate
// CHECK: linalg.reduce
// CHECK-LABEL: func.func @no_candidate
// CHECK: return

func.func @one_candidate(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2xf32>)
      -> tensor<2xf32>
  %result = linalg.reduce ins(%arg0 : tensor<2x4xf32>)
      outs(%init : tensor<2xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %result : tensor<2xf32>
}

func.func @no_candidate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
