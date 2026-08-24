// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" %S/../QGEMMRecognize/Inputs/q40_q80_canonical.ttshared.mlir | FileCheck %s
//
// Tiling behavior on the measured Q4_0 x Q8_0 capture. This file no longer
// tests recognition boundaries; the recognition input is owned by the sibling
// QGEMMRecognize corpus.

// CHECK-LABEL: func.func @q40_q80_canonical
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// CHECK-NOT: scf.for
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
