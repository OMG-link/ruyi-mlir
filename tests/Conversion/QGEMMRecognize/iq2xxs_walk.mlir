// RUN: buddy-opt %s > %t.no-pass
// RUN: buddy-opt --qgemm-recognize %s > %t.with-pass
// RUN: diff -u %t.no-pass %t.with-pass
// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// An IQ2-style fragment with both table-lookup generics. The table values are
// captured by tensor.extract inside the parallel-only regions; OperatorWalk
// must follow those captured tensor operands to their to_tensor boundaries.

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] op 0 walk: ops=9 boundaries=10
// CHECK-LABEL: func.func @iq2xxs_walk
// CHECK: bufferization.to_tensor
// CHECK: bufferization.to_tensor
// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK-NOT: qgemm.node
func.func @iq2xxs_walk(%codes: tensor<2x3x4xi32>,
                        %signs: tensor<2x3x4xi32>,
                        %codebook_memref: memref<4x16xf32>,
                        %sign_memref: memref<4x16xf32>) -> tensor<2x3xf32> {
  %zero = arith.constant 0.0 : f32
  %codebook = bufferization.to_tensor %codebook_memref restrict writable
      : memref<4x16xf32> to tensor<4x16xf32>
  %sign_table = bufferization.to_tensor %sign_memref restrict writable
      : memref<4x16xf32> to tensor<4x16xf32>

  %lookup0_empty = tensor.empty() : tensor<2x3x4xf32>
  %lookup0 = linalg.generic {
    indexing_maps = [#identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%codes : tensor<2x3x4xi32>)
      outs(%lookup0_empty : tensor<2x3x4xf32>) {
  ^bb0(%code: i32, %out: f32):
    %index = arith.index_cast %code : i32 to index
    %value = tensor.extract %codebook[%index, %index] : tensor<4x16xf32>
    linalg.yield %value : f32
  } -> tensor<2x3x4xf32>

  %lookup1_empty = tensor.empty() : tensor<2x3x4xf32>
  %lookup1 = linalg.generic {
    indexing_maps = [#identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%signs : tensor<2x3x4xi32>)
      outs(%lookup1_empty : tensor<2x3x4xf32>) {
  ^bb0(%sign: i32, %out: f32):
    %index = arith.index_cast %sign : i32 to index
    %value = tensor.extract %sign_table[%index, %index] : tensor<4x16xf32>
    linalg.yield %value : f32
  } -> tensor<2x3x4xf32>

  %product_empty = tensor.empty() : tensor<2x3x4xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lookup0, %lookup1 : tensor<2x3x4xf32>, tensor<2x3x4xf32>)
      outs(%product_empty : tensor<2x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<2x3x4xf32>
  %expanded = tensor.expand_shape %product [[0], [1], [2, 3]]
      output_shape [2, 3, 2, 2]
      : tensor<2x3x4xf32> into tensor<2x3x2x2xf32>
  %inner_empty = tensor.empty() : tensor<2x3x2xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
  %inner = linalg.reduce ins(%expanded : tensor<2x3x2x2xf32>)
      outs(%inner_init : tensor<2x3x2xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %scaled_empty = tensor.empty() : tensor<2x3x2xf32>
  %scaled = linalg.generic {
    indexing_maps = [#identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner : tensor<2x3x2xf32>)
      outs(%scaled_empty : tensor<2x3x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x3x2xf32>
  %final_empty = tensor.empty() : tensor<2x3xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<2x3xf32>) -> tensor<2x3xf32>
  %final = linalg.reduce ins(%scaled : tensor<2x3x2xf32>)
      outs(%final_init : tensor<2x3xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<2x3xf32>
}
