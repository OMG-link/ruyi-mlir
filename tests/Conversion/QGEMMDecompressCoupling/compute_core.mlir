// Property 7: compute core preservation (Requirement 5.6).
//
// For every capture under Inputs/ and both decompress values, the multiply,
// the downstream reductions and the final reduction (ops in V_m and V_n)
// keep their result shapes, hence m/n at their input positions, compared
// against the recognize-only output by Inputs/compute_core_check.py. The
// checker also compares every non-broadcast stage=compute op of the reference
// with the output by kind, indexing maps, body, attributes and types
// (Req 12.15); under pre it restricts that comparison to the reference compute
// ops in V_m ∩ V_n, because the recognizer's compute stage may include ops
// that vary along one parallel dim only, which pre rebuilds without the other.
//
// Recognize-only output passes this test by design: the compute core is what
// the pass must leave unchanged (Req 5.6), so identity preserves it too.
//
// CHECK-NOT: VIOLATION
// CHECK: COMPUTE-CORE OK {{[1-9][0-9]*}} {{[1-9][0-9]*}} {{[1-9][0-9]*}} {{[0-9]+}}
// --- q80_q80_strict
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.pre
// RUN: %python %S/Inputs/compute_core_check.py --mode=pre %t.q80_q80_strict.ref %t.q80_q80_strict.pre | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.on-the-fly
// RUN: %python %S/Inputs/compute_core_check.py --mode=on-the-fly %t.q80_q80_strict.ref %t.q80_q80_strict.on-the-fly | FileCheck %s
// --- q40_q80_canonical
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.pre
// RUN: %python %S/Inputs/compute_core_check.py --mode=pre %t.q40_q80_canonical.ref %t.q40_q80_canonical.pre | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.on-the-fly
// RUN: %python %S/Inputs/compute_core_check.py --mode=on-the-fly %t.q40_q80_canonical.ref %t.q40_q80_canonical.on-the-fly | FileCheck %s
// --- q4k_q8k_canonical
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.pre
// RUN: %python %S/Inputs/compute_core_check.py --mode=pre %t.q4k_q8k_canonical.ref %t.q4k_q8k_canonical.pre | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.on-the-fly
// RUN: %python %S/Inputs/compute_core_check.py --mode=on-the-fly %t.q4k_q8k_canonical.ref %t.q4k_q8k_canonical.on-the-fly | FileCheck %s
// --- iq2xxs_q8k_canonical
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.pre
// RUN: %python %S/Inputs/compute_core_check.py --mode=pre %t.iq2xxs_q8k_canonical.ref %t.iq2xxs_q8k_canonical.pre | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.on-the-fly
// RUN: %python %S/Inputs/compute_core_check.py --mode=on-the-fly %t.iq2xxs_q8k_canonical.ref %t.iq2xxs_q8k_canonical.on-the-fly | FileCheck %s
