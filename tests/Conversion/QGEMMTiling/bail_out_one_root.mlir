// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=16 tile-n=8" %S/../QGEMMRecognize/Inputs/iq2xxs_q8k_canonical.ttshared.mlir | FileCheck %s
//
// A normal single-root behavior test using the measured codebook capture.
// Both output dimensions are larger than the requested tiles.

// CHECK-LABEL: func.func @iq2xxs_q8k_canonical
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c16{{[_0-9]*}}
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c8{{[_0-9]*}}
// CHECK: qgemm.node = {root = 0 : i64
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
