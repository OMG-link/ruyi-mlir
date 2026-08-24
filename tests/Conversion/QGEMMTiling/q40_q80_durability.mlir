// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" --qgemm-tiling="tile-m=4 tile-n=4" %S/../QGEMMRecognize/Inputs/q40_q80_canonical.ttshared.mlir | tee %t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=Q40-DATA --input-file=%t %s
//
// The complete four-pass pipeline keeps all data preparation outside both
// tiling levels. The inner 4x4 tile contains the compute broadcast, multiply,
// reduction and annotated final root, while the destination write remains
// after the loop nest.

// Q40-DATA-LABEL: func.func @q40_q80_canonical
// Q40-DATA-COUNT-15: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// Q40-DATA-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// Q40-DATA: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// Q40-DATA-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// Q40-DATA: {{^    }}bufferization.materialize_in_destination
// Q40-DATA-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}

// CHECK-LABEL: func.func @q40_q80_canonical
// CHECK-COUNT-15: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// CHECK-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// CHECK-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// CHECK: {{^      }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
// CHECK: {{^          }}%{{[a-zA-Z0-9_]+}} = linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// CHECK: {{^          }}%{{[a-zA-Z0-9_]+}} = linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// CHECK: {{^            }}%{{[a-zA-Z0-9_]+}} = arith.muli
// CHECK: {{^          }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// CHECK: {{^          }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: {{^    }}bufferization.materialize_in_destination
// CHECK-NOT: qgemm.node = {root = 0 : i64, stage = "data_prep"}
