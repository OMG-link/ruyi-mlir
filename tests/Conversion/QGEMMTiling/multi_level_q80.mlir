// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir | tee %t.first | FileCheck %s --check-prefix=FIRST
// RUN: FileCheck --check-prefix=FIRST-NODES --input-file=%t.first %s
// RUN: FileCheck --check-prefix=FIRST-DATA --input-file=%t.first %s
// RUN: FileCheck --check-prefix=FIRST-COMPUTE --input-file=%t.first %s
// RUN: FileCheck --check-prefix=FIRST-ROOT --input-file=%t.first %s
// RUN: FileCheck --check-prefix=FIRST-FILLS --input-file=%t.first %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=TRACE
// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %t.first | tee %t.second | FileCheck %s --check-prefix=SECOND
// RUN: FileCheck --check-prefix=SECOND-NODES --input-file=%t.second %s
// RUN: FileCheck --check-prefix=SECOND-DATA --input-file=%t.second %s
// RUN: FileCheck --check-prefix=SECOND-COMPUTE --input-file=%t.second %s
// RUN: FileCheck --check-prefix=SECOND-ROOT --input-file=%t.second %s
// RUN: FileCheck --check-prefix=SECOND-FILLS --input-file=%t.second %s
//
// The first round deliberately includes the simplifier: without it this measured
// capture carries 20 qgemm nodes, while the canonical pipeline produces exactly
// 17 nodes and a walk trace of ops=17 boundaries=13 before recognition. The
// second round consumes %t.first and does not rerun recognition.

// TRACE: [qgemm-recognize] op 0 walk: ops=17 boundaries=13

// FIRST-LABEL: func.func @q80_q80_strict
// FIRST: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// FIRST-NOT: scf.for
// FIRST: {{^      }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root =
// FIRST-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// FIRST: {{^    }}bufferization.materialize_in_destination

// FIRST-NODES-LABEL: func.func @q80_q80_strict
// FIRST-NODES-COUNT-17: qgemm.node = {root =
// FIRST-NODES-NOT: qgemm.node = {root =
// FIRST-DATA-LABEL: func.func @q80_q80_strict
// FIRST-DATA-COUNT-6: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// FIRST-DATA-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// FIRST-DATA: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// FIRST-DATA-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// FIRST-COMPUTE-LABEL: func.func @q80_q80_strict
// FIRST-COMPUTE-COUNT-11: qgemm.node = {root = 0 : i64, stage = "compute"}
// FIRST-COMPUTE-NOT: qgemm.node = {root = 0 : i64, stage = "compute"}
// FIRST-ROOT-LABEL: func.func @q80_q80_strict
// FIRST-ROOT-COUNT-1: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// FIRST-ROOT-NOT: qgemm.root
// FIRST-FILLS-LABEL: func.func @q80_q80_strict
// FIRST-FILLS-NOT: linalg.fill
// FIRST-FILLS: {{^    }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : i32) outs(%{{[a-zA-Z0-9_]+}} : tensor<192x32x16xi32>) -> tensor<192x32x16xi32>{{[[:space:]]*$}}
// FIRST-FILLS-NOT: linalg.fill
// FIRST-FILLS: {{^      }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<32x32xf32>) -> tensor<32x32xf32>{{[[:space:]]*$}}
// FIRST-FILLS-NOT: linalg.fill

// SECOND-LABEL: func.func @q80_q80_strict
// SECOND: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// SECOND: {{^      }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// SECOND: {{^        }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// SECOND: {{^          }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root =
// SECOND-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// SECOND: {{^    }}bufferization.materialize_in_destination

// SECOND-NODES-LABEL: func.func @q80_q80_strict
// SECOND-NODES-COUNT-17: qgemm.node = {root =
// SECOND-NODES-NOT: qgemm.node = {root =
// SECOND-DATA-LABEL: func.func @q80_q80_strict
// SECOND-DATA-COUNT-6: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// SECOND-DATA-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// SECOND-DATA: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// SECOND-DATA-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// SECOND-COMPUTE-LABEL: func.func @q80_q80_strict
// SECOND-COMPUTE-COUNT-11: qgemm.node = {root = 0 : i64, stage = "compute"}
// SECOND-COMPUTE-NOT: qgemm.node = {root = 0 : i64, stage = "compute"}
// SECOND-ROOT-LABEL: func.func @q80_q80_strict
// SECOND-ROOT-COUNT-1: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// SECOND-ROOT-NOT: qgemm.root
// SECOND-FILLS-LABEL: func.func @q80_q80_strict
// SECOND-FILLS-NOT: linalg.fill
// SECOND-FILLS: {{^    }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : i32) outs(%{{[a-zA-Z0-9_]+}} : tensor<192x32x16xi32>) -> tensor<192x32x16xi32>{{[[:space:]]*$}}
// SECOND-FILLS-NOT: linalg.fill
// SECOND-FILLS: {{^          }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// SECOND-FILLS-NOT: linalg.fill
