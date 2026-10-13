// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80_q80_strict.ttshared.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q80_q80_strict.ttshared.mlir | %python %S/Inputs/scope_check.py %t.q80_q80_strict.ttshared.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q80_q80_strict.ttshared.mlir | %python %S/Inputs/scope_check.py %t.q80_q80_strict.ttshared.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40_q80_canonical.ttshared.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q40_q80_canonical.ttshared.mlir | %python %S/Inputs/scope_check.py %t.q40_q80_canonical.ttshared.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q40_q80_canonical.ttshared.mlir | %python %S/Inputs/scope_check.py %t.q40_q80_canonical.ttshared.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k_q8k_canonical.ttshared.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir | %python %S/Inputs/scope_check.py %t.q4k_q8k_canonical.ttshared.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/q4k_q8k_canonical.ttshared.mlir | %python %S/Inputs/scope_check.py %t.q4k_q8k_canonical.ttshared.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2xxs_q8k_canonical.ttshared.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir | %python %S/Inputs/scope_check.py %t.iq2xxs_q8k_canonical.ttshared.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir | %python %S/Inputs/scope_check.py %t.iq2xxs_q8k_canonical.ttshared.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/scope_outside_user.mlir -o %t.scope_outside_user.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/scope_outside_user.mlir | %python %S/Inputs/scope_check.py %t.scope_outside_user.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %S/Inputs/scope_outside_user.mlir | %python %S/Inputs/scope_check.py %t.scope_outside_user.ref | FileCheck %s
//
// Property 11 (scope preservation) and Property 12 (broadcast uniqueness)
// over every capture in Inputs/ and both modes, plus the hand-written
// Inputs/scope_outside_user.mlir, whose dequantized weight factor is also
// returned from the function (an Outside_User in the middle of a chain).
// Inputs/scope_check.py compares the pass output against the recognize-only
// output, both in generic form with SSA names erased, and checks that
//  - ops outside every scope other than tensor.empty/linalg.fill/constants
//    are unchanged (kind, operand and result types, attributes, regions),
//    which includes the operand types seen by every Outside_User;
//  - no tensor.empty/linalg.fill/constant and no annotated op has zero users;
//  - no two Pure_Broadcasts share the same input and the same input map.
// CHECK-NOT: VIOLATION
// CHECK: SCOPE OK {{[1-9][0-9]*}} {{[0-9]+}}
