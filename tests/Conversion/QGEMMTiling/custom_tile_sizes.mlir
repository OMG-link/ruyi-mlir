// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=16 tile-n=8" %S/../QGEMMRecognize/Inputs/q40_q80_canonical.ttshared.mlir | FileCheck %s
//
// Non-default tile sizes on the measured Q4_0 x Q8_0 capture exercise both
// output dimensions: M=192 uses 16 and N=32 uses 8.

// CHECK-LABEL: func.func @q40_q80_canonical
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c16{{[_0-9]*}}
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c8{{[_0-9]*}}
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
