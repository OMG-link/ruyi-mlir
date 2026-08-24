// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=256 tile-n=64" %S/../QGEMMRecognize/Inputs/q4k_q8k_canonical.ttshared.mlir | FileCheck %s
//
// A second no-tiling case on a measured two-level capture. Both tile sizes
// exceed the 192x32 result dimensions.

// CHECK-LABEL: func.func @q4k_q8k_canonical
// CHECK-NOT: scf.for
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
