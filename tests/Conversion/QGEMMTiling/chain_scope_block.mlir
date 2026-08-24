// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" %S/../QGEMMRecognize/Inputs/q4k_q8k_canonical.ttshared.mlir | FileCheck %s
//
// Tiling behavior on the measured Q4_K x Q8_K capture, including its
// multi-level reduction chain.

// CHECK-LABEL: func.func @q4k_q8k_canonical
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// CHECK-NOT: scf.for
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
