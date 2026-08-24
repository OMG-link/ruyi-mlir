// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir | FileCheck %s
//
// With the measured Q8_0 x Q8_0 output shape 192x32, only M is larger than
// the requested 32x32 tile.

// CHECK-LABEL: func.func @q80_q80_strict
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// CHECK-NOT: scf.for
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
