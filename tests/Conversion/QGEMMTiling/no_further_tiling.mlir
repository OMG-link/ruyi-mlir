// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=192 tile-n=32" %S/../QGEMMRecognize/Inputs/q40_q80_canonical.ttshared.mlir | FileCheck %s
//
// A measured capture whose 192x32 result is already no larger than the
// requested tiles must retain its annotation without introducing loops.

// CHECK-LABEL: func.func @q40_q80_canonical
// CHECK-NOT: scf.for
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
