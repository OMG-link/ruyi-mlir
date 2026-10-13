// Property 3: input-shape independence (Requirement 3.3).
//
// The three inputs under Inputs/shape_independence_*.mlir describe the same
// Q4_0-style operator with the same function name and differ only in where
// the weight's broadcast along m sits:
//   a: folded into the multiply's indexing map, decompression in (n,k) space;
//   b: an explicit pure broadcast before the decompression chain;
//   c: folded into the decompressing (computing) generic's input map.
// Separation yields the same G(r) for all three, so in each mode the outputs
// must be Topologically_Identical, checked with Inputs/topo_equal.py.
//
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/shape_independence_a_multiply_map.mlir -o %t.pre.a
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/shape_independence_b_early_broadcast.mlir -o %t.pre.b
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/shape_independence_c_computing_generic.mlir -o %t.pre.c
// RUN: %python %S/Inputs/topo_equal.py %t.pre.a %t.pre.b
// RUN: %python %S/Inputs/topo_equal.py %t.pre.a %t.pre.c
//
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/shape_independence_a_multiply_map.mlir -o %t.otf.a
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/shape_independence_b_early_broadcast.mlir -o %t.otf.b
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/shape_independence_c_computing_generic.mlir -o %t.otf.c
// RUN: %python %S/Inputs/topo_equal.py %t.otf.a %t.otf.b
// RUN: %python %S/Inputs/topo_equal.py %t.otf.a %t.otf.c
//
// Guard against a vacuous pass: the two modes must differ, and the output
// must still carry the annotated root.
// RUN: %python %S/Inputs/topo_equal.py --expect-different %t.pre.a %t.otf.a
// RUN: FileCheck %s --input-file=%t.pre.a
// RUN: FileCheck %s --input-file=%t.otf.a
// CHECK-LABEL: sym_name = "qgemm"
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
