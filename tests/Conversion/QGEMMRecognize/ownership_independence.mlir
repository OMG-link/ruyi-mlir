// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %s > %t.default 2> %t.default.stderr
// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize="explain=true" %s > %t.explain 2> %t.explain.stderr
// RUN: buddy-opt --mlir-disable-threading %s > %t.input
// RUN: FileCheck %s --check-prefix=DEFAULT-DIAG < %t.default.stderr
// RUN: FileCheck %s --check-prefix=DEFAULT-TRACE --implicit-check-not=shared-non-boundary-operation < %t.default.stderr
// RUN: FileCheck %s --check-prefix=EXPLAIN-DIAG < %t.explain.stderr
// RUN: FileCheck %s --check-prefix=EXPLAIN-TRACE --implicit-check-not=shared-non-boundary-operation < %t.explain.stderr
// RUN: FileCheck %s --check-prefix=DEFAULT-IR < %t.default

// The first function has two valid candidates that read the same
// bufferization.to_tensor operation.  A shared upstream boundary is not
// ownership sharing; both products use that one operation's result.
// The second function combines one rejected candidate with an independent
// valid candidate; the valid one must still be accepted.

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#broadcast_both = affine_map<(d0, d1, d2) -> (0, 0, d2)>

// DEFAULT-DIAG-NOT: error:
// DEFAULT-DIAG-NOT: warning:
// DEFAULT-DIAG-NOT: remark:
// DEFAULT-DIAG-NOT: note:
// EXPLAIN-DIAG-COUNT-1: remark: qgemm-recognize: rejected candidate: reason=final-result-rank section=4
// EXPLAIN-DIAG-NOT: remark: qgemm-recognize: rejected candidate:
// EXPLAIN-DIAG-NOT: error:
// EXPLAIN-DIAG-NOT: warning:
// DEFAULT-TRACE-NOT: shared-non-boundary-operation

// DEFAULT-TRACE: [qgemm-recognize] candidates: 2
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 0 accepted: root=0
// DEFAULT-TRACE: [qgemm-recognize] op 1 accepted: root=1
// DEFAULT-TRACE: [qgemm-recognize] candidates: 2
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] candidate 0 rejected: reason=final-result-rank section=4 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("ownership_independence.mlir":220:1)
// DEFAULT-TRACE: [qgemm-recognize] op 1 accepted: root=0

// EXPLAIN-TRACE-NOT: shared-non-boundary-operation
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 2
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 0 accepted: root=0
// EXPLAIN-TRACE: [qgemm-recognize] op 1 accepted: root=1
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 2
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] candidate 0 rejected: reason=final-result-rank section=4 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("ownership_independence.mlir":220:1)
// EXPLAIN-TRACE: [qgemm-recognize] op 1 accepted: root=0

func.func @shared_upstream_boundary(%shared_memref: memref<4x1x4xf32>,
                                    %rhs_a: tensor<1x4x4xf32>,
                                    %rhs_b: tensor<1x4x4xf32>) -> tensor<4x4xf32> {
  %shared_lhs = bufferization.to_tensor %shared_memref restrict writable
      : memref<4x1x4xf32> to tensor<4x1x4xf32>
  %product_a_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_a = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared_lhs, %rhs_a : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_a_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %zero_a = arith.constant 0.0 : f32
  %final_a_empty = tensor.empty() : tensor<4x4xf32>
  %final_a_init = linalg.fill ins(%zero_a : f32)
      outs(%final_a_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_a = linalg.reduce ins(%product_a : tensor<4x4x4xf32>)
      outs(%final_a_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %product_b_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_b = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared_lhs, %rhs_b : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_b_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %zero_b = arith.constant 0.0 : f32
  %final_b_empty = tensor.empty() : tensor<4x4xf32>
  %final_b_init = linalg.fill ins(%zero_b : f32)
      outs(%final_b_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_b = linalg.reduce ins(%product_b : tensor<4x4x4xf32>)
      outs(%final_b_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final_a : tensor<4x4xf32>
}

// DEFAULT-IR-LABEL: func.func @shared_upstream_boundary
// The shared bufferization boundary is unannotated, while both consumers
// receive their own complete root identity.
// DEFAULT-IR: bufferization.to_tensor %{{.*}} restrict writable : memref<4x1x4xf32> to tensor<4x1x4xf32>{{ *$}}
// DEFAULT-IR: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// DEFAULT-IR: linalg.generic{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @rejected_and_independent_valid(%bad_lhs: tensor<4x1x4xf32>,
                                           %bad_rhs: tensor<1x4x4xf32>,
                                           %good_lhs: tensor<4x4xf32>,
                                           %good_rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // These reshape and broadcast operations belong only to the rejected
  // candidate, so they must remain free of both annotation attributes.
  %bad_reshape = tensor.expand_shape %bad_lhs [[0], [1], [2, 3]]
      output_shape [4, 1, 2, 2]
      : tensor<4x1x4xf32> into tensor<4x1x2x2xf32>
  %bad_roundtrip = tensor.collapse_shape %bad_reshape [[0], [1], [2, 3]]
      : tensor<4x1x2x2xf32> into tensor<4x1x4xf32>
  %bad_broadcast_empty = tensor.empty() : tensor<4x4x4xf32>
  %bad_broadcast = linalg.generic {
    indexing_maps = [#broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bad_rhs : tensor<1x4x4xf32>)
      outs(%bad_broadcast_empty : tensor<4x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x4xf32>
  %bad_product_empty = tensor.empty() : tensor<4x4x4xf32>
  %bad_product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bad_roundtrip, %bad_broadcast : tensor<4x1x4xf32>, tensor<4x4x4xf32>)
      outs(%bad_product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %bad_zero = arith.constant 0.0 : f32
  %bad_final_empty = tensor.empty() : tensor<4xf32>
  %bad_final_init = linalg.fill ins(%bad_zero : f32)
      outs(%bad_final_empty : tensor<4xf32>) -> tensor<4xf32>
  %bad_final = linalg.reduce ins(%bad_product : tensor<4x4x4xf32>)
      outs(%bad_final_init : tensor<4xf32>) dimensions = [1, 2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("ownership_independence.mlir":220:1)

  %good_lhs_3d = tensor.expand_shape %good_lhs [[0], [1, 2]]
      output_shape [4, 1, 4]
      : tensor<4x4xf32> into tensor<4x1x4xf32>
  %good_rhs_3d = tensor.expand_shape %good_rhs [[0, 1], [2]]
      output_shape [1, 4, 4]
      : tensor<4x4xf32> into tensor<1x4x4xf32>
  %good_product_empty = tensor.empty() : tensor<4x4x4xf32>
  %good_product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%good_lhs_3d, %good_rhs_3d : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%good_product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %good_zero = arith.constant 0.0 : f32
  %good_final_empty = tensor.empty() : tensor<4x4xf32>
  %good_final_init = linalg.fill ins(%good_zero : f32)
      outs(%good_final_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %good_final = linalg.reduce ins(%good_product : tensor<4x4x4xf32>)
      outs(%good_final_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %good_final : tensor<4x4xf32>
}

// DEFAULT-IR-LABEL: func.func @rejected_and_independent_valid
// Every operation unique to the rejected scope is printed without either
// annotation attribute: reshape, broadcast, product and final reduction.
// DEFAULT-IR: tensor.expand_shape %{{.*}} {{.*}} output_shape [4, 1, 2, 2] : tensor<4x1x4xf32> into tensor<4x1x2x2xf32>{{ *$}}
// DEFAULT-IR: tensor.collapse_shape %{{.*}} {{.*}} : tensor<4x1x2x2xf32> into tensor<4x1x4xf32>{{ *$}}
// DEFAULT-IR: linalg.generic{{.*}}ins(%{{.*}} : tensor<1x4x4xf32>) outs(%{{.*}} : tensor<4x4x4xf32>) {
// DEFAULT-IR: linalg.generic{{.*}}ins(%{{.*}}, %{{.*}} : tensor<4x1x4xf32>, tensor<4x4x4xf32>) outs(%{{.*}} : tensor<4x4x4xf32>) {
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [1, 2]{{ *}}{{ *$}}
// The independent accepted candidate has both stages and a complete root.
// DEFAULT-IR: tensor.expand_shape{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// DEFAULT-IR: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

// DEFAULT-IR-LABEL: func.func @shared_tensor_empty_boundary
// The shared tensor.empty destination is root-neutral; both accepted roots
// remain independently annotated.
// DEFAULT-IR: tensor.empty() : tensor<4x4x4xf32>{{ *$}}
// DEFAULT-IR: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// DEFAULT-IR: linalg.generic{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

// The next function shares only a tensor.empty destination between two
// otherwise independent products.  The shared empty must remain root-neutral.
// DEFAULT-TRACE: [qgemm-recognize] candidates: 2
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 0 accepted: root=0
// DEFAULT-TRACE: [qgemm-recognize] op 1 accepted: root=1
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 2
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 0 accepted: root=0
// EXPLAIN-TRACE: [qgemm-recognize] op 1 accepted: root=1
func.func @shared_tensor_empty_boundary(%lhs_a: tensor<4x1x4xf32>,
                                        %rhs_a: tensor<1x4x4xf32>,
                                        %lhs_b: tensor<4x1x4xf32>,
                                        %rhs_b: tensor<1x4x4xf32>)
    -> tensor<4x4xf32> {
  %shared_product_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_a = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs_a, %rhs_a : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%shared_product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %product_b = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs_b, %rhs_b : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%shared_product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %zero = arith.constant 0.0 : f32
  %final_a_empty = tensor.empty() : tensor<4x4xf32>
  %final_a_init = linalg.fill ins(%zero : f32)
      outs(%final_a_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_a = linalg.reduce ins(%product_a : tensor<4x4x4xf32>)
      outs(%final_a_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %final_b_empty = tensor.empty() : tensor<4x4xf32>
  %final_b_init = linalg.fill ins(%zero : f32)
      outs(%final_b_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_b = linalg.reduce ins(%product_b : tensor<4x4x4xf32>)
      outs(%final_b_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final_a : tensor<4x4xf32>
}

// This function shares a pure-tensor fill result between two final
// reductions.  Traversal must stop at the fill and must not claim its empty
// destination for either root.
// DEFAULT-TRACE: [qgemm-recognize] candidates: 2
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 0 accepted: root=0
// DEFAULT-TRACE: [qgemm-recognize] op 1 accepted: root=1
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 2
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 0 accepted: root=0
// EXPLAIN-TRACE: [qgemm-recognize] op 1 accepted: root=1
// DEFAULT-IR-LABEL: func.func @shared_tensor_fill_boundary
// The shared tensor-semantics fill is unannotated and both final reductions
// retain their own roots.
// DEFAULT-IR: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{ *$}}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @shared_tensor_fill_boundary(%lhs_a: tensor<4x1x4xf32>,
                                       %rhs_a: tensor<1x4x4xf32>,
                                       %lhs_b: tensor<4x1x4xf32>,
                                       %rhs_b: tensor<1x4x4xf32>)
    -> tensor<4x4xf32> {
  %product_a_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_a = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs_a, %rhs_a : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_a_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %product_b_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_b = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs_b, %rhs_b : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_b_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %zero = arith.constant 0.0 : f32
  %shared_empty = tensor.empty() : tensor<4x4xf32>
  %shared_init = linalg.fill ins(%zero : f32)
      outs(%shared_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_a = linalg.reduce ins(%product_a : tensor<4x4x4xf32>)
      outs(%shared_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %final_b = linalg.reduce ins(%product_b : tensor<4x4x4xf32>)
      outs(%shared_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final_a : tensor<4x4xf32>
}

// This function shares a tensor-valued ConstantLike producer between two
// products.  The boundary rule must be trait-based rather than arith.constant
// specific so that the shared constant is not root-owned.
// DEFAULT-TRACE: [qgemm-recognize] candidates: 2
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 0 accepted: root=0
// DEFAULT-TRACE: [qgemm-recognize] op 1 accepted: root=1
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 2
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 0 accepted: root=0
// EXPLAIN-TRACE: [qgemm-recognize] op 1 accepted: root=1
// DEFAULT-IR-LABEL: func.func @shared_tensor_constant_like_boundary
// The shared tensor ConstantLike producer is root-neutral; the two products
// and their final reductions still carry distinct roots.
// DEFAULT-IR: arith.constant dense<1.000000e+00> : tensor<4x1x4xf32>{{ *$}}
// DEFAULT-IR: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// DEFAULT-IR: linalg.generic{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// DEFAULT-IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @shared_tensor_constant_like_boundary(%rhs_a: tensor<1x4x4xf32>,
                                                %rhs_b: tensor<1x4x4xf32>)
    -> tensor<4x4xf32> {
  %shared = arith.constant dense<1.0> : tensor<4x1x4xf32>
  %product_a_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_a = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared, %rhs_a : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_a_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %product_b_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_b = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared, %rhs_b : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_b_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %zero = arith.constant 0.0 : f32
  %final_a_empty = tensor.empty() : tensor<4x4xf32>
  %final_a_init = linalg.fill ins(%zero : f32)
      outs(%final_a_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_a = linalg.reduce ins(%product_a : tensor<4x4x4xf32>)
      outs(%final_a_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %final_b_empty = tensor.empty() : tensor<4x4xf32>
  %final_b_init = linalg.fill ins(%zero : f32)
      outs(%final_b_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_b = linalg.reduce ins(%product_b : tensor<4x4x4xf32>)
      outs(%final_b_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final_a : tensor<4x4xf32>
}
