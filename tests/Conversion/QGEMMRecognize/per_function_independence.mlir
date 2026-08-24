// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %s > %t.ir 2> %t.trace
// RUN: FileCheck %s --check-prefix=IR < %t.ir
// RUN: FileCheck %s --check-prefix=TRACE < %t.trace

// Candidate numbering restarts for every function.  Both functions contain
// an accepted operator, and each function's non-root and final operations
// carry root 0 independently of the other function.

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>

// IR-LABEL: func.func @first_candidate
// IR: tensor.expand_shape{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// IR: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// IR-NOT: qgemm.root
// IR-LABEL: func.func @second_candidate
// IR: tensor.expand_shape{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// IR: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// IR-NOT: qgemm.root

// TRACE: [qgemm-recognize] candidates: 1
// TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// TRACE: [qgemm-recognize] op 0 accepted: root=0
// TRACE: [qgemm-recognize] candidates: 1
// TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// TRACE: [qgemm-recognize] op 0 accepted: root=0

func.func @first_candidate(%lhs: tensor<8x32xf32>,
                            %rhs: tensor<8x32xf32>,
                            %factor: tensor<8x8x1xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %lhs_3d = tensor.expand_shape %lhs [[0], [1, 2]]
      output_shape [8, 1, 32]
      : tensor<8x32xf32> into tensor<8x1x32xf32>
  %rhs_3d = tensor.expand_shape %rhs [[0, 1], [2]]
      output_shape [1, 8, 32]
      : tensor<8x32xf32> into tensor<1x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs_3d, %rhs_3d : tensor<8x1x32xf32>, tensor<1x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
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
  ^bb0(%value: f32, %scale: f32, %out: f32):
    %scaled_value = arith.mulf %value, %scale : f32
    linalg.yield %scaled_value : f32
  } -> tensor<8x8x1xf32>

  %final_empty = tensor.empty() : tensor<8x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %final = linalg.reduce ins(%scaled : tensor<8x8x1xf32>)
      outs(%final_init : tensor<8x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<8x8xf32>
}

func.func @second_candidate(%lhs: tensor<8x32xf32>,
                             %rhs: tensor<8x32xf32>,
                             %factor: tensor<8x8x1xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %lhs_3d = tensor.expand_shape %lhs [[0], [1, 2]]
      output_shape [8, 1, 32]
      : tensor<8x32xf32> into tensor<8x1x32xf32>
  %rhs_3d = tensor.expand_shape %rhs [[0, 1], [2]]
      output_shape [1, 8, 32]
      : tensor<8x32xf32> into tensor<1x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs_3d, %rhs_3d : tensor<8x1x32xf32>, tensor<1x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
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
  ^bb0(%value: f32, %scale: f32, %out: f32):
    %scaled_value = arith.mulf %value, %scale : f32
    linalg.yield %scaled_value : f32
  } -> tensor<8x8x1xf32>

  %final_empty = tensor.empty() : tensor<8x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %final = linalg.reduce ins(%scaled : tensor<8x8x1xf32>)
      outs(%final_init : tensor<8x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<8x8xf32>
}
