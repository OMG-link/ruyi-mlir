// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" %S/../QGEMMRecognize/Inputs/iq2xxs_q8k_canonical.ttshared.mlir | FileCheck %s
//
// The measured iq2_xxs x Q8_K capture includes codebook and sign lookup
// generics in the recognized scope. Tiling must consume their annotation and
// produce the same single M loop as the other one-level output shapes.

// CHECK-LABEL: func.func @iq2xxs_q8k_canonical
// CHECK: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// CHECK-NOT: scf.for
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK: bufferization.materialize_in_destination
