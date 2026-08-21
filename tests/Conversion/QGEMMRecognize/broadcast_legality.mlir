// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 >/dev/null | FileCheck %s
//
// These candidates all have a rank-2 final reduction.  They are rejected by
// the core broadcast condition for different reasons:
//   * same_core_dim has two single-dimension edges, both for output dim 0;
//   * dataprep_broadcast has a dim-0 core edge followed by a dim-1 edge in
//     data_prep;
//   * multidim_broadcast has one {0, 1} edge and one single dim-0 edge.
// The final reduction in multidim_broadcast also checks the rejected trace's
// operation, SSA result and explicit source location.

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast_dim0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#broadcast_dim1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_both = affine_map<(d0, d1, d2) -> (0, 0, d2)>

// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("{{.*}}broadcast_legality.mlir":{{[0-9]+}}:12)
// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("{{.*}}broadcast_legality.mlir":{{[0-9]+}}:12)
// CHECK: [qgemm-recognize] candidates: 1
// CHECK: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("broadcast_legality.mlir":123:3)
// CHECK-NOT: accepted:

func.func @same_core_dim(%lhs: tensor<1x8x32xf32>,
                          %rhs: tensor<1x8x32xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32

  %lhs_empty = tensor.empty() : tensor<8x8x32xf32>
  %lhs_broadcast = linalg.generic {
    indexing_maps = [#broadcast_dim0, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs : tensor<1x8x32xf32>)
      outs(%lhs_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %rhs_empty = tensor.empty() : tensor<8x8x32xf32>
  %rhs_broadcast = linalg.generic {
    indexing_maps = [#broadcast_dim0, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%rhs : tensor<1x8x32xf32>)
      outs(%rhs_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs_broadcast, %rhs_broadcast : tensor<8x8x32xf32>, tensor<8x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
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

func.func @dataprep_broadcast(%source: tensor<1x1x32xf32>,
                              %plain: tensor<8x8x32xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32

  %inner_empty = tensor.empty() : tensor<1x8x32xf32>
  %inner_broadcast = linalg.generic {
    indexing_maps = [#broadcast_dim1, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<1x1x32xf32>)
      outs(%inner_empty : tensor<1x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x8x32xf32>

  %outer_empty = tensor.empty() : tensor<8x8x32xf32>
  %outer_broadcast = linalg.generic {
    indexing_maps = [#broadcast_dim0, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner_broadcast : tensor<1x8x32xf32>)
      outs(%outer_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%outer_broadcast, %plain : tensor<8x8x32xf32>, tensor<8x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
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

func.func @multidim_broadcast(%source: tensor<1x1x32xf32>,
                              %dim0_source: tensor<1x8x32xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32

  %both_empty = tensor.empty() : tensor<8x8x32xf32>
  %both_broadcast = linalg.generic {
    indexing_maps = [#broadcast_both, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<1x1x32xf32>)
      outs(%both_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %dim0_empty = tensor.empty() : tensor<8x8x32xf32>
  %dim0_broadcast = linalg.generic {
    indexing_maps = [#broadcast_dim0, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%dim0_source : tensor<1x8x32xf32>)
      outs(%dim0_empty : tensor<8x8x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x32xf32>

  %product_empty = tensor.empty() : tensor<8x8x32xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%both_broadcast, %dim0_broadcast : tensor<8x8x32xf32>, tensor<8x8x32xf32>)
      outs(%product_empty : tensor<8x8x32xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
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
    } loc("broadcast_legality.mlir": 123:3)
  return %final : tensor<8x8xf32>
}
