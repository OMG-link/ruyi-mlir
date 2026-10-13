// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q80_q80_strict_noalias.ttshared.mlir | %python %S/Inputs/check_pre_invariant.py | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir | %python %S/Inputs/check_pre_invariant.py | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir | %python %S/Inputs/check_pre_invariant.py | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir | %python %S/Inputs/check_pre_invariant.py | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict_noalias.ttshared.mlir | not %python %S/Inputs/check_pre_invariant.py | FileCheck %s --check-prefix=OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir | not %python %S/Inputs/check_pre_invariant.py | FileCheck %s --check-prefix=OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir | not %python %S/Inputs/check_pre_invariant.py | FileCheck %s --check-prefix=OTFQ4K
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir | not %python %S/Inputs/check_pre_invariant.py | FileCheck %s --check-prefix=OTF
//
// Property 5 (`pre` invariant) over every capture in Inputs/.
// Inputs/check_pre_invariant.py parses the pass output and verifies that
//  - no annotated linalg.generic other than a Pure_Broadcast introduces m or n
//    (an ins operand map omitting a parallel m/n dim that the output map has);
//  - every m/n Pure_Broadcast is used, and each broadcast chain ends at a
//    same-root op that carries the inserted dim (a V_d op) or at an
//    Outside_User (another root, an unannotated op, or the terminator).
//  - no op performs decompression computation between an m/n broadcast and
//    the multiply: the first non-broadcast, non-dim-transform op reading a
//    broadcast value as data must join it with a value that does not come
//    from a broadcast of the same dim (Req 12.14).
//  - no annotated memref.reinterpret_cast has a stride-0 dimension: that is
//    a parallel dimension its operand does not carry, introduced by an op
//    that is not a Pure_Broadcast, which only on-the-fly creates (Req 7.9).
// The script prints one VIOLATION line per failure, then a summary line.
//
// The on-the-fly output of every capture (whose two modes differ) must fail
// the invariant. `not` alone would also accept a crash, so the OTF runs
// require the stride-0 VIOLATION line and the FAIL summary the script prints
// only after a complete analysis. On-the-fly folds every parallel dimension
// of these captures into a load view, so its output holds no pure broadcast
// at all, which the trailing 0 pins down; the pre output has four or more.
// CHECK-NOT: VIOLATION
// CHECK: PRE-INVARIANT OK {{[0-9]+}} {{[1-9][0-9]*}}
// OTF: VIOLATION: {{.*}}introduces a parallel dimension at stride 0
// OTF: PRE-INVARIANT FAIL {{[0-9]+}} 0
// On Q4_K two of the three %arg5 scale loads keep their copy (Req 14.7 (3)),
// cannot take the stride-0 m dim, and keep their m broadcast.
// OTFQ4K: VIOLATION: {{.*}}introduces a parallel dimension at stride 0
// OTFQ4K: PRE-INVARIANT FAIL {{[0-9]+}} 2
