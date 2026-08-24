// RUN: buddy-opt --mlir-disable-threading --qgemm-recognize %s > %t.once
// RUN: buddy-opt --mlir-disable-threading --qgemm-recognize --qgemm-recognize %s > %t.twice
// RUN: diff -u %t.once %t.twice
// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %s > %t.first.ir 2> %t.first.trace && buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %t.first.ir > %t.second.ir 2> %t.second.trace && diff -u %t.first.ir %t.second.ir && diff -u %t.first.trace %t.second.trace
// RUN: buddy-opt --mlir-disable-threading %t.first.ir > %t.roundtrip
// RUN: diff -u %t.first.ir %t.roundtrip
// RUN: FileCheck %s --check-prefix=CHECK < %t.first.ir
// RUN: FileCheck %s --check-prefix=ROOT < %t.first.ir
// RUN: FileCheck %s --check-prefix=TRACE < %t.first.trace

// The two reductions below are independent operators.  They deliberately
// share a ConstantLike input, a tensor.empty destination, and tensor-semantics
// linalg.fill results.  Those values stay root-neutral while each root-owned
// operation receives only its own identity.  The anchored boundary checks use
// end-of-line matching so an added qgemm.node or qgemm.root would fail them.

#lhs_broadcast = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs_broadcast3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @annotation_contract
// ConstantLike tensor result and scalar plumbing are root-neutral.
// CHECK: arith.constant dense<1.000000e+00> : tensor<4x32xf32>{{ *$}}
// CHECK: arith.constant 0.000000e+00 : f32{{ *$}}
// Data-preparation nodes are annotated, while their source values are not.
// CHECK: tensor.expand_shape %{{.*}} {{.*}} output_shape [1, 4, 32] {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// CHECK: tensor.expand_shape %{{.*}} {{.*}} output_shape [1, 4, 32] {qgemm.node = {root = 1 : i64, stage = "data_prep"}}
// The shared product destination stays root-neutral.
// CHECK: tensor.empty() : tensor<4x4x32xf32>{{ *$}}
// The first product proves pre-existing attributes survive annotation.
// CHECK: linalg.generic{{.*}}broadcastDims = array<i64: 1>, custom = "keep-me", qgemm.node = {root = 0 : i64, stage = "compute"}
// CHECK: linalg.generic{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}
// Non-root compute operations carry qgemm.node but no qgemm.root.
// CHECK: tensor.expand_shape %{{.*}} {{.*}} output_shape [4, 4, 1, 32] {qgemm.node = {root = 0 : i64, stage = "compute"}}
// CHECK: tensor.expand_shape %{{.*}} {{.*}} output_shape [4, 4, 1, 32] {qgemm.node = {root = 1 : i64, stage = "compute"}}
// The shared inner empty and fill are both root-neutral boundaries.
// CHECK: tensor.empty() : tensor<4x4x1xf32>{{ *$}}
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x4x1xf32>) -> tensor<4x4x1xf32>{{ *$}}
// CHECK: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// CHECK: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}
// The shared final destination and fill have neither annotation.
// CHECK: tensor.empty() : tensor<4x4xf32>{{ *$}}
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{ *$}}
// Each final reduction is the only root-bearing operation for its operator.
// CHECK: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: return

// ROOT-LABEL: func.func @annotation_contract
// ROOT-NOT: qgemm.root
// ROOT: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// ROOT-NOT: qgemm.root
// ROOT: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// ROOT-NOT: qgemm.root

// TRACE: [qgemm-recognize] candidates: 2
// TRACE: [qgemm-recognize] op 0 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// TRACE: [qgemm-recognize] op 1 walk: ops={{[0-9]+}} boundaries={{[0-9]+}}
// TRACE: [qgemm-recognize] op 0 accepted: root=0
// TRACE: [qgemm-recognize] op 1 accepted: root=1

func.func @annotation_contract(%rhs_a: tensor<4x32xf32>,
                               %rhs_b: tensor<4x32xf32>) -> tensor<4x4xf32> {
  %shared_lhs = arith.constant dense<1.0> : tensor<4x32xf32>
  %zero = arith.constant 0.0 : f32
  %rhs_a_3d = tensor.expand_shape %rhs_a [[0, 1], [2]]
      output_shape [1, 4, 32]
      : tensor<4x32xf32> into tensor<1x4x32xf32>
  %rhs_b_3d = tensor.expand_shape %rhs_b [[0, 1], [2]]
      output_shape [1, 4, 32]
      : tensor<4x32xf32> into tensor<1x4x32xf32>

  %shared_product_empty = tensor.empty() : tensor<4x4x32xf32>
  %product_a = linalg.generic {
    indexing_maps = [#lhs_broadcast, #rhs_broadcast3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared_lhs, %rhs_a_3d : tensor<4x32xf32>, tensor<1x4x32xf32>)
      outs(%shared_product_empty : tensor<4x4x32xf32>) attrs = {broadcastDims = array<i64: 1>, custom = "keep-me"} {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<4x4x32xf32>
  %product_b = linalg.generic {
    indexing_maps = [#lhs_broadcast, #rhs_broadcast3, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared_lhs, %rhs_b_3d : tensor<4x32xf32>, tensor<1x4x32xf32>)
      outs(%shared_product_empty : tensor<4x4x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %value = arith.mulf %lhs, %rhs : f32
    linalg.yield %value : f32
  } -> tensor<4x4x32xf32>

  %split_a = tensor.expand_shape %product_a [[0], [1], [2, 3]]
      output_shape [4, 4, 1, 32]
      : tensor<4x4x32xf32> into tensor<4x4x1x32xf32>
  %split_b = tensor.expand_shape %product_b [[0], [1], [2, 3]]
      output_shape [4, 4, 1, 32]
      : tensor<4x4x32xf32> into tensor<4x4x1x32xf32>

  %inner_empty = tensor.empty() : tensor<4x4x1xf32>
  %inner_init = linalg.fill ins(%zero : f32)
      outs(%inner_empty : tensor<4x4x1xf32>) -> tensor<4x4x1xf32>
  %inner_a = linalg.reduce ins(%split_a : tensor<4x4x1x32xf32>)
      outs(%inner_init : tensor<4x4x1xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %inner_b = linalg.reduce ins(%split_b : tensor<4x4x1x32xf32>)
      outs(%inner_init : tensor<4x4x1xf32>) dimensions = [3]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %final_empty = tensor.empty() : tensor<4x4xf32>
  %final_init = linalg.fill ins(%zero : f32)
      outs(%final_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_a = linalg.reduce ins(%inner_a : tensor<4x4x1xf32>)
      outs(%final_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %final_b = linalg.reduce ins(%inner_b : tensor<4x4x1xf32>)
      outs(%final_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %final_a : tensor<4x4xf32>
}
