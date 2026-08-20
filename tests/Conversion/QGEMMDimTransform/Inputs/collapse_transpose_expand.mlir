#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast_m = affine_map<(d0, d1, d2) -> (d0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (d1, d2)>

func.func @collapse_transpose_expand(%arg0: tensor<4x4xf32>,
                                      %arg1: tensor<6x4xf32>,
                                      %factor: tensor<4x6x2xf32>)
    -> tensor<4x6xf32> {
  %zero = arith.constant 0.0 : f32

  %product_empty = tensor.empty() : tensor<4x6x4xf32>
  %product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<6x4xf32>)
      outs(%product_empty : tensor<4x6x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<4x6x4xf32>

  %split = tensor.expand_shape %product [[0], [1], [2, 3]]
      output_shape [4, 6, 2, 2]
      : tensor<4x6x4xf32> into tensor<4x6x2x2xf32>
  %inner_empty = tensor.empty() : tensor<4x6x2xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<4x6x2xf32>) -> tensor<4x6x2xf32>
  %inner = linalg.reduce ins(%split : tensor<4x6x2x2xf32>)
      outs(%inner_init : tensor<4x6x2xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %scaled_empty = tensor.empty() : tensor<4x6x2xf32>
  %scaled = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner, %factor : tensor<4x6x2xf32>, tensor<4x6x2xf32>)
      outs(%scaled_empty : tensor<4x6x2xf32>) {
  ^bb0(%in: f32, %scale: f32, %out: f32):
    %value = arith.mulf %in, %scale : f32
    linalg.yield %value : f32
  } -> tensor<4x6x2xf32>

  %final_source = tensor.empty() : tensor<2x3x4xf32>
  %final_collapsed = tensor.collapse_shape %final_source [[0, 1], [2]]
      : tensor<2x3x4xf32> into tensor<6x4xf32>
  %final_transpose_empty = tensor.empty() : tensor<4x6xf32>
  %final_transposed = linalg.transpose ins(%final_collapsed : tensor<6x4xf32>)
      outs(%final_transpose_empty : tensor<4x6xf32>) permutation = [1, 0]
  %final_expanded = tensor.expand_shape %final_transposed [[0, 1], [2]]
      output_shape [2, 2, 6]
      : tensor<4x6xf32> into tensor<2x2x6xf32>
  %final_destination = tensor.empty() : tensor<2x2x7xf32>
  %final_inserted = tensor.insert_slice %final_expanded into %final_destination[0, 0, 0]
      [2, 2, 6] [1, 1, 1]
      : tensor<2x2x6xf32> into tensor<2x2x7xf32>
  %final_extracted = tensor.extract_slice %final_inserted[0, 0, 1]
      [2, 2, 6] [1, 1, 1]
      : tensor<2x2x7xf32> to tensor<2x2x6xf32>
  %final_out_seed = tensor.collapse_shape %final_extracted [[0, 1], [2]]
      : tensor<2x2x6xf32> into tensor<4x6xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_out_seed : tensor<4x6xf32>) -> tensor<4x6xf32>
  %final = linalg.reduce ins(%scaled : tensor<4x6x2xf32>)
      outs(%final_init : tensor<4x6xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<4x6xf32>
}
