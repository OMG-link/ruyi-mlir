// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q80_q80_strict.ttshared.mlir > %t.q80.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict.ttshared.mlir | %python %S/Inputs/annotation_check.py %t.q80.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q80_q80_strict.ttshared.mlir | %python %S/Inputs/annotation_check.py %t.q80.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q40_q80_canonical.ttshared.mlir > %t.q40.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q40_q80_canonical.ttshared.mlir | %python %S/Inputs/annotation_check.py %t.q40.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q40_q80_canonical.ttshared.mlir | %python %S/Inputs/annotation_check.py %t.q40.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q4k_q8k_canonical.ttshared.mlir > %t.q4k.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q4k_q8k_canonical.ttshared.mlir | %python %S/Inputs/annotation_check.py %t.q4k.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q4k_q8k_canonical.ttshared.mlir | %python %S/Inputs/annotation_check.py %t.q4k.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir > %t.iq2.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir | %python %S/Inputs/annotation_check.py %t.iq2.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir | %python %S/Inputs/annotation_check.py %t.iq2.ref | FileCheck %s
//
// Property 10 (annotation completeness) over every capture in Inputs/, under
// both decompress modes. Inputs/annotation_check.py compares the pass output
// against the recognizer-only output and verifies that
//  - every qgemm.node names a reference root and has stage data_prep/compute;
//  - stage is compute iff the op's result carries both the m and n axes
//    (traced structurally backwards from m_dim / n_dim of qgemm.root through
//    indexing maps and shape ops) or the op is an m/n Pure_Broadcast, and
//    data_prep otherwise;
//  - no tensor.empty / linalg.fill / arith.constant carries qgemm.node;
//  - each root keeps exactly one qgemm.root with the original m_dim / n_dim.
// CHECK-NOT: VIOLATION
// CHECK: ANNOTATION OK {{[1-9][0-9]*}} {{[0-9]+}}
//
// Negative assertion (Req 12.9-12.13): the stage rule above also holds on the
// recognize-only output of q80, q40 and q4k, so it alone cannot tell that the
// pass rewrote anything. Where the pass rewrites (every pair except the
// Req 6.1 skips: pre on q80/q40, on-the-fly on iq2xxs) its output must differ
// from the reference.
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.ref.g
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.on-the-fly.g
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q80_q80_strict.ref.g %t.q80_q80_strict.on-the-fly.g
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.ref.g
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.on-the-fly.g
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q40_q80_canonical.ref.g %t.q40_q80_canonical.on-the-fly.g
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.ref.g
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.pre.g
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q4k_q8k_canonical.ref.g %t.q4k_q8k_canonical.pre.g
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.on-the-fly.g
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q4k_q8k_canonical.ref.g %t.q4k_q8k_canonical.on-the-fly.g
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.ref.g
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.pre.g
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.iq2xxs_q8k_canonical.ref.g %t.iq2xxs_q8k_canonical.pre.g
