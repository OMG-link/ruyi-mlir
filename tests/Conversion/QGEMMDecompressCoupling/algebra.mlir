// Algebraic properties of --qgemm-decompress-coupling over every capture in
// Inputs/, compared with the Topology_Checker (Inputs/topo_equal.py) on
// generic-form output, so op order within a block and SSA names are ignored.
//
// x is the recognized capture (simplify + recognize). The coupling pass is
// re-run directly on its own outputs, because the recognizer is never re-run
// on transformed IR.
//
// Property 1 (idempotence, Req 9.1, 6.1, 12.5): P_A(P_A(x)) ~ P_A(x), on the
// recognized capture and on the pass's own composed outputs.
//
// Property 2 (placement commutativity across modes) is removed with Req 9.2
// and this file asserts nothing about it. Under on-the-fly a parallel
// dimension is folded into a stride-0 memref.reinterpret_cast view (Req 7.9),
// and a zero stride is not visible in the result type, so the form is not
// reversible: pre on such output is a no-op. The file keeps idempotence and
// the negative assertions below (Req 12.5, 12.11, 12.12).

// --- q80_q80_strict
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.pre
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.pre -o %t.q80_q80_strict.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q80_q80_strict.pre %t.q80_q80_strict.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.otf -o %t.q80_q80_strict.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q80_q80_strict.otf %t.q80_q80_strict.otf.otf
// Composed output y = P_pre(P_otf(x)), where P_pre takes the Req 6.1 skip
// on x: idempotence of each mode on y.
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.otf -o %t.q80_q80_strict.ypre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.ypre -o %t.q80_q80_strict.ypre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.ypre -o %t.q80_q80_strict.ypre.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.ypre.pre -o %t.q80_q80_strict.ypre.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q80_q80_strict.ypre.pre %t.q80_q80_strict.ypre.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.ypre.otf -o %t.q80_q80_strict.ypre.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q80_q80_strict.ypre.otf %t.q80_q80_strict.ypre.otf.otf
// Composed outputs of y = P_otf(x): idempotence of each mode on y.
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.otf -o %t.q80_q80_strict.otf.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.otf -o %t.q80_q80_strict.otf.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.otf.pre -o %t.q80_q80_strict.otf.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q80_q80_strict.otf.pre %t.q80_q80_strict.otf.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q80_q80_strict.otf.otf -o %t.q80_q80_strict.otf.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q80_q80_strict.otf.otf %t.q80_q80_strict.otf.otf.otf

// --- q40_q80_canonical
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.pre
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.pre -o %t.q40_q80_canonical.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q40_q80_canonical.pre %t.q40_q80_canonical.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.otf -o %t.q40_q80_canonical.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q40_q80_canonical.otf %t.q40_q80_canonical.otf.otf
// Composed output y = P_pre(P_otf(x)), where P_pre takes the Req 6.1 skip
// on x: idempotence of each mode on y.
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.otf -o %t.q40_q80_canonical.ypre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.ypre -o %t.q40_q80_canonical.ypre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.ypre -o %t.q40_q80_canonical.ypre.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.ypre.pre -o %t.q40_q80_canonical.ypre.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q40_q80_canonical.ypre.pre %t.q40_q80_canonical.ypre.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.ypre.otf -o %t.q40_q80_canonical.ypre.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q40_q80_canonical.ypre.otf %t.q40_q80_canonical.ypre.otf.otf
// Composed outputs of y = P_otf(x): idempotence of each mode on y.
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.otf -o %t.q40_q80_canonical.otf.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.otf -o %t.q40_q80_canonical.otf.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.otf.pre -o %t.q40_q80_canonical.otf.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q40_q80_canonical.otf.pre %t.q40_q80_canonical.otf.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q40_q80_canonical.otf.otf -o %t.q40_q80_canonical.otf.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q40_q80_canonical.otf.otf %t.q40_q80_canonical.otf.otf.otf

// --- q4k_q8k_canonical
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.pre
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.pre -o %t.q4k_q8k_canonical.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q4k_q8k_canonical.pre %t.q4k_q8k_canonical.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.otf -o %t.q4k_q8k_canonical.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q4k_q8k_canonical.otf %t.q4k_q8k_canonical.otf.otf
// Composed outputs of y = P_pre(x): idempotence of each mode on y.
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.pre -o %t.q4k_q8k_canonical.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.pre -o %t.q4k_q8k_canonical.pre.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.pre.pre -o %t.q4k_q8k_canonical.pre.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q4k_q8k_canonical.pre.pre %t.q4k_q8k_canonical.pre.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.pre.otf -o %t.q4k_q8k_canonical.pre.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q4k_q8k_canonical.pre.otf %t.q4k_q8k_canonical.pre.otf.otf
// Composed outputs of y = P_otf(x): idempotence of each mode on y.
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.otf -o %t.q4k_q8k_canonical.otf.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.otf -o %t.q4k_q8k_canonical.otf.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.otf.pre -o %t.q4k_q8k_canonical.otf.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.q4k_q8k_canonical.otf.pre %t.q4k_q8k_canonical.otf.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.q4k_q8k_canonical.otf.otf -o %t.q4k_q8k_canonical.otf.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.q4k_q8k_canonical.otf.otf %t.q4k_q8k_canonical.otf.otf.otf

// --- iq2xxs_q8k_canonical
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.pre
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.pre -o %t.iq2xxs_q8k_canonical.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.iq2xxs_q8k_canonical.pre %t.iq2xxs_q8k_canonical.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.otf -o %t.iq2xxs_q8k_canonical.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.iq2xxs_q8k_canonical.otf %t.iq2xxs_q8k_canonical.otf.otf
// Composed outputs of y = P_pre(x): idempotence of each mode on y.
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.pre -o %t.iq2xxs_q8k_canonical.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.pre -o %t.iq2xxs_q8k_canonical.pre.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.pre.pre -o %t.iq2xxs_q8k_canonical.pre.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.iq2xxs_q8k_canonical.pre.pre %t.iq2xxs_q8k_canonical.pre.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.pre.otf -o %t.iq2xxs_q8k_canonical.pre.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.iq2xxs_q8k_canonical.pre.otf %t.iq2xxs_q8k_canonical.pre.otf.otf
// Composed output y = P_otf(P_pre(x)), where P_otf takes the Req 6.1 skip
// on x: idempotence of each mode on y.
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.pre -o %t.iq2xxs_q8k_canonical.yotf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.yotf -o %t.iq2xxs_q8k_canonical.yotf.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.yotf -o %t.iq2xxs_q8k_canonical.yotf.otf
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.yotf.pre -o %t.iq2xxs_q8k_canonical.yotf.pre.pre
// RUN: %python %S/Inputs/topo_equal.py %t.iq2xxs_q8k_canonical.yotf.pre %t.iq2xxs_q8k_canonical.yotf.pre.pre
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %t.iq2xxs_q8k_canonical.yotf.otf -o %t.iq2xxs_q8k_canonical.yotf.otf.otf
// RUN: %python %S/Inputs/topo_equal.py %t.iq2xxs_q8k_canonical.yotf.otf %t.iq2xxs_q8k_canonical.yotf.otf.otf

// Discrimination (Req 12.9-12.13): the two modes differ on every capture, and
// the pass output differs from the recognize-only output x wherever the
// pass rewrites. Req 6.1 makes P_pre(x) == x on q80 and q40 and
// P_otf(x) == x on iq2xxs, so those pairs are not asserted to differ.
// --- q80_q80_strict: discrimination
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.rec
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q80_q80_strict.pre %t.q80_q80_strict.otf
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q80_q80_strict.rec %t.q80_q80_strict.otf
// --- q40_q80_canonical: discrimination
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.rec
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q40_q80_canonical.pre %t.q40_q80_canonical.otf
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q40_q80_canonical.rec %t.q40_q80_canonical.otf
// --- q4k_q8k_canonical: discrimination
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.rec
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q4k_q8k_canonical.pre %t.q4k_q8k_canonical.otf
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q4k_q8k_canonical.rec %t.q4k_q8k_canonical.pre
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.q4k_q8k_canonical.rec %t.q4k_q8k_canonical.otf
// --- iq2xxs_q8k_canonical: discrimination
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.rec
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.iq2xxs_q8k_canonical.pre %t.iq2xxs_q8k_canonical.otf
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.iq2xxs_q8k_canonical.rec %t.iq2xxs_q8k_canonical.pre
