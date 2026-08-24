// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir | FileCheck %s
//
// Single-round tiling of the measured Q8_0 x Q8_0 capture. The 192x32
// result gets one M loop; N is exactly the requested tile size.

// CHECK-LABEL: func.func @q80_q80_strict
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// CHECK-NOT: scf.for
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
