// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %s > %t.default 2> %t.default.stderr
// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize="explain=true" %s > %t.explain 2> %t.explain.stderr
// RUN: buddy-opt --mlir-disable-threading %s > %t.input
// RUN: diff -u %t.input %t.default
// RUN: diff -u %t.input %t.explain
// RUN: FileCheck %s --check-prefix=DEFAULT-IR < %t.default
// RUN: FileCheck %s --check-prefix=DEFAULT-DIAG < %t.default.stderr
// RUN: FileCheck %s --check-prefix=DEFAULT-TRACE < %t.default.stderr
// RUN: FileCheck %s --check-prefix=EXPLAIN-DIAG < %t.explain.stderr
// RUN: FileCheck %s --check-prefix=EXPLAIN-TOTAL < %t.explain.stderr
// RUN: FileCheck %s --check-prefix=EXPLAIN-TRACE < %t.explain.stderr

// Every function below contains a real additive reduction candidate.  The
// default run must be silent and immutable; the explain run reports exactly
// one remark for each rejected candidate while retaining the same trace.

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#broadcast_both = affine_map<(d0, d1, d2) -> (0, 0, d2)>

// DEFAULT-IR: module {
// DEFAULT-IR-NOT: qgemm.node
// DEFAULT-IR-NOT: qgemm.root
// DEFAULT-DIAG-NOT: error:
// DEFAULT-DIAG-NOT: warning:
// DEFAULT-DIAG-NOT: remark:
// DEFAULT-DIAG-NOT: note:

// The rank check is first, so this rank-1 final result is rejected before
// stage or broadcast checks are considered.
// DEFAULT-TRACE: [qgemm-recognize] candidates: 1
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] candidate 0 rejected: reason=final-result-rank section=4 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":100:1)
// EXPLAIN-TOTAL-COUNT-8: remark: qgemm-recognize: rejected candidate:
// EXPLAIN-TOTAL-NOT: remark: qgemm-recognize: rejected candidate:
// EXPLAIN-DIAG-COUNT-1: negative_rejections.mlir:100:1: remark: qgemm-recognize: rejected candidate: reason=final-result-rank section=4
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 1
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] candidate 0 rejected: reason=final-result-rank section=4 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":100:1)

func.func @final_result_rank(%lhs: tensor<4x1x4xf32>,
                             %rhs: tensor<1x4x4xf32>) -> tensor<4xf32> {
  %zero = arith.constant 0.0 : f32
  %product_empty = tensor.empty() : tensor<4x4x4xf32>
  %product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %final_empty = tensor.empty() : tensor<4xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<4xf32>) -> tensor<4xf32>
  %final = linalg.reduce ins(%product : tensor<4x4x4xf32>)
      outs(%final_init : tensor<4xf32>) dimensions = [1, 2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("negative_rejections.mlir":100:1)
  return %final : tensor<4xf32>
}

// Both traversal orders reach %shared once through a direct compute edge and
// once through a broadcast edge.  The rejection must be stage-conflict in
// either order, and the explain remark is still attached to the final reduce.
// DEFAULT-TRACE: [qgemm-recognize] candidates: 1
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] candidate 0 rejected: reason=stage-conflict section=2 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":110:1)
// DEFAULT-TRACE: [qgemm-recognize] candidates: 1
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] candidate 0 rejected: reason=stage-conflict section=2 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":120:1)
// EXPLAIN-DIAG-COUNT-1: negative_rejections.mlir:111:1: remark: qgemm-recognize: rejected candidate: reason=stage-conflict section=2
// EXPLAIN-DIAG-COUNT-1: negative_rejections.mlir:121:1: remark: qgemm-recognize: rejected candidate: reason=stage-conflict section=2
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 1
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] candidate 0 rejected: reason=stage-conflict section=2 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":110:1)
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 1
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] candidate 0 rejected: reason=stage-conflict section=2 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":120:1)

func.func @stage_conflict_compute_first(%source: tensor<4x1x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %shared_empty = tensor.empty() : tensor<4x4x4xf32>
  %shared = linalg.generic {
    indexing_maps = [#broadcast_m, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<4x1x4xf32>)
      outs(%shared_empty : tensor<4x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x4xf32> loc("negative_rejections.mlir":110:1)
  %broadcast_empty = tensor.empty() : tensor<4x4x4xf32>
  %broadcast = linalg.generic {
    indexing_maps = [#broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared : tensor<4x4x4xf32>)
      outs(%broadcast_empty : tensor<4x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x4xf32>
  %product_empty = tensor.empty() : tensor<4x4x4xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared, %broadcast : tensor<4x4x4xf32>, tensor<4x4x4xf32>)
      outs(%product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %final_empty = tensor.empty() : tensor<4x4xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final = linalg.reduce ins(%product : tensor<4x4x4xf32>)
      outs(%final_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("negative_rejections.mlir":111:1)
  return %final : tensor<4x4xf32>
}

func.func @stage_conflict_data_prep_first(%source: tensor<4x1x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %shared_empty = tensor.empty() : tensor<4x4x4xf32>
  %shared = linalg.generic {
    indexing_maps = [#broadcast_m, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<4x1x4xf32>)
      outs(%shared_empty : tensor<4x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x4xf32> loc("negative_rejections.mlir":120:1)
  %broadcast_empty = tensor.empty() : tensor<4x4x4xf32>
  %broadcast = linalg.generic {
    indexing_maps = [#broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared : tensor<4x4x4xf32>)
      outs(%broadcast_empty : tensor<4x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x4xf32>
  %product_empty = tensor.empty() : tensor<4x4x4xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%broadcast, %shared : tensor<4x4x4xf32>, tensor<4x4x4xf32>)
      outs(%product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %final_empty = tensor.empty() : tensor<4x4xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final = linalg.reduce ins(%product : tensor<4x4x4xf32>)
      outs(%final_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("negative_rejections.mlir":121:1)
  return %final : tensor<4x4xf32>
}

// The next three candidates fail only the single-dimension M/N coverage
// condition: two edges for the same dimension, the missing direction only in
// data_prep, and one multi-dimension edge respectively.
// DEFAULT-TRACE: [qgemm-recognize] candidates: 1
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":130:1)
// DEFAULT-TRACE: [qgemm-recognize] candidates: 1
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":131:1)
// DEFAULT-TRACE: [qgemm-recognize] candidates: 1
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":132:1)
// EXPLAIN-DIAG-COUNT-1: negative_rejections.mlir:130:1: remark: qgemm-recognize: rejected candidate: reason=core-broadcast-dimensions section=2
// EXPLAIN-DIAG-COUNT-1: negative_rejections.mlir:131:1: remark: qgemm-recognize: rejected candidate: reason=core-broadcast-dimensions section=2
// EXPLAIN-DIAG-COUNT-1: negative_rejections.mlir:132:1: remark: qgemm-recognize: rejected candidate: reason=core-broadcast-dimensions section=2
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 1
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":130:1)
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 1
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":131:1)
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 1
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] candidate 0 rejected: reason=core-broadcast-dimensions section=2 at=linalg.reduce value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":132:1)

func.func @core_same_dimension(%lhs: tensor<1x4x4xf32>,
                               %rhs: tensor<1x4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %product_empty = tensor.empty() : tensor<4x4x4xf32>
  %product = linalg.generic {
    indexing_maps = [#broadcast_n, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<1x4x4xf32>, tensor<1x4x4xf32>)
      outs(%product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %final_empty = tensor.empty() : tensor<4x4xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final = linalg.reduce ins(%product : tensor<4x4x4xf32>)
      outs(%final_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("negative_rejections.mlir":130:1)
  return %final : tensor<4x4xf32>
}

func.func @core_missing_direction_in_data_prep(%source: tensor<1x1x4xf32>,
                                               %plain: tensor<4x4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %inner_empty = tensor.empty() : tensor<1x4x4xf32>
  %inner = linalg.generic {
    indexing_maps = [#broadcast_m, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<1x1x4xf32>)
      outs(%inner_empty : tensor<1x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x4x4xf32>
  %outer_empty = tensor.empty() : tensor<4x4x4xf32>
  %outer = linalg.generic {
    indexing_maps = [#broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%inner : tensor<1x4x4xf32>)
      outs(%outer_empty : tensor<4x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x4xf32>
  %product_empty = tensor.empty() : tensor<4x4x4xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%outer, %plain : tensor<4x4x4xf32>, tensor<4x4x4xf32>)
      outs(%product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %final_empty = tensor.empty() : tensor<4x4xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final = linalg.reduce ins(%product : tensor<4x4x4xf32>)
      outs(%final_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("negative_rejections.mlir":131:1)
  return %final : tensor<4x4xf32>
}

func.func @core_multi_dimension_edge(%source: tensor<1x1x4xf32>,
                                     %dim0_source: tensor<1x4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %both_empty = tensor.empty() : tensor<4x4x4xf32>
  %both = linalg.generic {
    indexing_maps = [#broadcast_both, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<1x1x4xf32>)
      outs(%both_empty : tensor<4x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x4xf32>
  %dim0_empty = tensor.empty() : tensor<4x4x4xf32>
  %dim0 = linalg.generic {
    indexing_maps = [#broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%dim0_source : tensor<1x4x4xf32>)
      outs(%dim0_empty : tensor<4x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4x4xf32>
  %product_empty = tensor.empty() : tensor<4x4x4xf32>
  %product = linalg.generic {
    indexing_maps = [#identity3, #identity3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%both, %dim0 : tensor<4x4x4xf32>, tensor<4x4x4xf32>)
      outs(%product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %final_empty = tensor.empty() : tensor<4x4xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final = linalg.reduce ins(%product : tensor<4x4x4xf32>)
      outs(%final_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("negative_rejections.mlir":132:1)
  return %final : tensor<4x4xf32>
}

// Both roots use the same product operation, which is not an upstream
// boundary.  Ownership arbitration therefore rejects both after their
// individual rank/stage/broadcast checks succeed.
// DEFAULT-TRACE: [qgemm-recognize] candidates: 2
// DEFAULT-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// DEFAULT-TRACE: [qgemm-recognize] candidate 0 rejected: reason=shared-non-boundary-operation section=13 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":140:1)
// DEFAULT-TRACE: [qgemm-recognize] candidate 1 rejected: reason=shared-non-boundary-operation section=13 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":140:1)
// EXPLAIN-DIAG-COUNT-1: negative_rejections.mlir:141:1: remark: qgemm-recognize: rejected candidate: reason=shared-non-boundary-operation section=13
// EXPLAIN-DIAG-COUNT-1: negative_rejections.mlir:142:1: remark: qgemm-recognize: rejected candidate: reason=shared-non-boundary-operation section=13
// EXPLAIN-TRACE: [qgemm-recognize] candidates: 2
// EXPLAIN-TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// EXPLAIN-TRACE: [qgemm-recognize] candidate 0 rejected: reason=shared-non-boundary-operation section=13 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":140:1)
// EXPLAIN-TRACE: [qgemm-recognize] candidate 1 rejected: reason=shared-non-boundary-operation section=13 at=linalg.generic value=%{{[A-Za-z0-9_]+}} loc=loc("negative_rejections.mlir":140:1)

func.func @shared_non_boundary(%lhs: tensor<4x1x4xf32>,
                              %rhs: tensor<1x4x4xf32>) -> tensor<4x4xf32> {
  %product_empty = tensor.empty() : tensor<4x4x4xf32>
  %product = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32> loc("negative_rejections.mlir":140:1)
  %zero_a = arith.constant 0.0 : f32
  %final_a_empty = tensor.empty() : tensor<4x4xf32>
  %final_a_init = linalg.fill ins(%zero_a : f32)
      outs(%final_a_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_a = linalg.reduce ins(%product : tensor<4x4x4xf32>)
      outs(%final_a_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("negative_rejections.mlir":141:1)
  %zero_b = arith.constant 0.0 : f32
  %final_b_empty = tensor.empty() : tensor<4x4xf32>
  %final_b_init = linalg.fill ins(%zero_b : f32)
      outs(%final_b_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_b = linalg.reduce ins(%product : tensor<4x4x4xf32>)
      outs(%final_b_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } loc("negative_rejections.mlir":142:1)
  return %final_a : tensor<4x4xf32>
}
