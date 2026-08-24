// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4 explain" %s 2>&1 | FileCheck %s
//
// A malformed root annotation without qgemm.node is not recoverable: tiling
// skips it silently. A root with parseable qgemm.root dimensions and a
// parseable qgemm.node.root remains recoverable even when its stage metadata is
// absent, non-string, unknown, or data_prep. The legal compute-stage root is
// also kept as a no-warning control.

// Four recoverable roots warn once each, while the malformed and compute-stage
// roots do not warn. Diagnostics are not gated by explain.
// CHECK-COUNT-4: qgemm-tiling: tiling root does not carry stage = "compute"; continuing from qgemm.root
// CHECK-NOT: qgemm-tiling: tiling root does not carry stage = "compute"; continuing from qgemm.root
// CHECK-NOT: error:

// CHECK-LABEL: func.func @malformed_root
// CHECK-NOT: scf.for
// CHECK: linalg.reduce
// CHECK-SAME: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

// A valid root identity with data_prep stage is still tiled after one warning.
// CHECK-LABEL: func.func @data_prep_stage_root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: linalg.reduce{{.*}}qgemm.node = {root = 10 : i64, stage = "data_prep"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}

// An unknown stage is recoverable from qgemm.root and qgemm.node.root.
// CHECK-LABEL: func.func @unknown_stage_root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: linalg.reduce{{.*}}qgemm.node = {root = 11 : i64, stage = "unknown"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}

// A non-StringAttr stage is also recoverable and warns instead of crashing.
// CHECK-LABEL: func.func @non_string_stage_root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: linalg.reduce{{.*}}qgemm.node = {root = 12 : i64, stage = 7 : i64}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}

// A missing stage field does not make an otherwise located root disappear.
// CHECK-LABEL: func.func @missing_stage_root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: linalg.reduce{{.*}}qgemm.node = {root = 13 : i64}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}

// A valid compute stage remains silent and is tiled normally.
// CHECK-LABEL: func.func @compute_stage_root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: linalg.reduce{{.*}}qgemm.node = {root = 14 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @malformed_root(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

func.func @data_prep_stage_root(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 10 : i64, stage = "data_prep"},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

func.func @unknown_stage_root(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 11 : i64, stage = "unknown"},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

func.func @non_string_stage_root(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 12 : i64, stage = 7 : i64},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

func.func @missing_stage_root(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 13 : i64},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

func.func @compute_stage_root(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 14 : i64, stage = "compute"},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}
