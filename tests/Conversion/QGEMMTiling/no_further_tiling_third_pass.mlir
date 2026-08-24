// RUN: buddy-opt --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir | tee %t.first | FileCheck %s --check-prefix=FIRST
// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %t.first | tee %t.second | FileCheck %s --check-prefix=SECOND
// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %t.second | tee %t.third | FileCheck %s --check-prefix=THIRD
// RUN: diff -u %t.second %t.third
//
// The third invocation consumes %t.second and must be byte-identical to it.
// The first two invocations establish the outer and inner tiling levels.

// FIRST-LABEL: func.func @q80_q80_strict
// FIRST: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// FIRST-NOT: scf.for
// FIRST: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

// SECOND-LABEL: func.func @q80_q80_strict
// SECOND: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// SECOND: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// SECOND: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// SECOND: qgemm.node = {root = 0 : i64
// SECOND: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

// THIRD-LABEL: func.func @q80_q80_strict
// THIRD: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c192{{[_0-9]*}} step %c32{{[_0-9]*}}
// THIRD: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// THIRD: scf.for %{{[a-z0-9_]+}} = %c0{{[_0-9]*}} to %c32{{[_0-9]*}} step %c4{{[_0-9]*}}
// THIRD: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
