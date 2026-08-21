// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 >/dev/null | FileCheck %s
//
// The shared producer is reached once through a direct compute path and once
// through a broadcast path.  The two functions reverse the product operands,
// so both traversal orders must reject the candidate at the shared producer.

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast_dim0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#broadcast_dim1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] candidate 0 rejected: reason=stage-conflict section=2 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("stage_conflict.mlir":42:7)
// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] candidate 0 rejected: reason=stage-conflict section=2 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("stage_conflict.mlir":142:7)
// CHECK-NOT: accepted:

func.func @compute_first(%source: tensor<8x1x32xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32

  %shared_empty = tensor.empty() : tensor<8x8x32xf32>
  %shared = linalg.generic {
    indexing_maps = [#broadcast_dim1, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<8x1x32xf32>)
      outs(%shared_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32> loc("stage_conflict.mlir":42:7)

  %broadcast_empty = tensor.empty() : tensor<8x8x32xf32>
  %broadcast = linalg.generic {
    indexing_maps = [#broadcast_dim0, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared : tensor<8x8x32xf32>)
      outs(%broadcast_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared, %broadcast : tensor<8x8x32xf32>, tensor<8x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<8x8x32xf32>

  %final_empty = tensor.empty() : tensor<8x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %final = linalg.reduce ins(%product : tensor<8x8x32xf32>)
      outs(%final_init : tensor<8x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<8x8xf32>
}

func.func @data_prep_first(%source: tensor<8x1x32xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32

  %shared_empty = tensor.empty() : tensor<8x8x32xf32>
  %shared = linalg.generic {
    indexing_maps = [#broadcast_dim1, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<8x1x32xf32>)
      outs(%shared_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32> loc("stage_conflict.mlir":142:7)

  %broadcast_empty = tensor.empty() : tensor<8x8x32xf32>
  %broadcast = linalg.generic {
    indexing_maps = [#broadcast_dim0, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared : tensor<8x8x32xf32>)
      outs(%broadcast_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%broadcast, %shared : tensor<8x8x32xf32>, tensor<8x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<8x8x32xf32>

  %final_empty = tensor.empty() : tensor<8x8xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %final = linalg.reduce ins(%product : tensor<8x8x32xf32>)
      outs(%final_init : tensor<8x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final : tensor<8x8xf32>
}
