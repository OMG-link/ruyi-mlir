// RUN: buddy-opt %s > %t.no-pass
// RUN: buddy-opt --qgemm-recognize %s > %t.with-pass
// RUN: diff -u %t.no-pass %t.with-pass
// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// A Q8_0-shaped canonical fragment has an inner quantization-block reduction
// followed by the factor-free outer reduction. Only the outer reduction is a
// candidate root.

#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#broadcast_m = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#broadcast_n = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK-LABEL: func.func @q80_one_candidate
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK-NOT: qgemm.node
func.func @q80_one_candidate(%arg0: tensor<8x1x2x32xf32>,
                              %arg1: tensor<1x8x2x32xf32>) -> tensor<8x8xf32> {
  %empty0 = tensor.empty() : tensor<8x8x2x32xf32>
  %broadcast0 = linalg.generic {
    indexing_maps = [#broadcast_m, #identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<8x1x2x32xf32>) outs(%empty0 : tensor<8x8x2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x2x32xf32>

  %empty1 = tensor.empty() : tensor<8x8x2x32xf32>
  %broadcast1 = linalg.generic {
    indexing_maps = [#broadcast_n, #identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<1x8x2x32xf32>) outs(%empty1 : tensor<8x8x2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x2x32xf32>

  %empty2 = tensor.empty() : tensor<8x8x2x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity4, #identity4, #identity4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%broadcast0, %broadcast1 : tensor<8x8x2x32xf32>, tensor<8x8x2x32xf32>)
    outs(%empty2 : tensor<8x8x2x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<8x8x2x32xf32>

  %zero = arith.constant 0.0 : f32
  %empty3 = tensor.empty() : tensor<8x8x2xf32>
  %init3 = linalg.fill ins(%zero : f32) outs(%empty3 : tensor<8x8x2xf32>)
      -> tensor<8x8x2xf32>
  %inner = linalg.reduce ins(%product : tensor<8x8x2x32xf32>)
      outs(%init3 : tensor<8x8x2xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %empty4 = tensor.empty() : tensor<8x8x2xf32>
  %scaled = linalg.generic {
    indexing_maps = [#identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner : tensor<8x8x2xf32>) outs(%empty4 : tensor<8x8x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x2xf32>

  %empty5 = tensor.empty() : tensor<8x8xf32>
  %init5 = linalg.fill ins(%zero : f32) outs(%empty5 : tensor<8x8xf32>)
      -> tensor<8x8xf32>
  %outer = linalg.reduce ins(%scaled : tensor<8x8x2xf32>)
      outs(%init5 : tensor<8x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %outer : tensor<8x8xf32>
}
