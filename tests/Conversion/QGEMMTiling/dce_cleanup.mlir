// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=4 tile-n=4" %S/../QGEMMRecognize/Inputs/q4k_q8k_canonical.ttshared.mlir | FileCheck %s
//
// The measured two-level Q4_K x Q8_K capture is tiled to 4x4 register-sized
// output tiles and remains connected to its destination write.

// CHECK-LABEL: func.func @q4k_q8k_canonical
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
