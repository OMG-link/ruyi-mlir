// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q80_q80_strict_noalias.ttshared.mlir -o %t.q80.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q80_q80_strict_noalias.ttshared.mlir | %python %S/Inputs/load_chain_check.py --forwarded %t.q80.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir -o %t.q40.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir | %python %S/Inputs/load_chain_check.py --forwarded %t.q40.ref | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir -o %t.q4k.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir | not %python %S/Inputs/load_chain_check.py --forwarded %t.q4k.ref | FileCheck %s --check-prefix=Q4K
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir -o %t.iq2.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir | %python %S/Inputs/load_chain_check.py --forwarded %t.iq2.ref | FileCheck %s
//
// Req 14.7, 14.9 / 12.21: under `pre` Load_Copy_Forwarding still drops every
// operand load chain's private copy, on every capture: no annotated
// memref.alloc or memref.copy is left, each bufferization.to_tensor reads its
// memref view op directly without `writable`, and otherwise the chains are
// the recognizer's: the view ops and the to_tensor result types match the
// recognize-only output exactly, so no chain op gains a parallel dimension.
// The comparison has to be chain-local: on q4k and iq2xxs the `pre` output
// differs from the recognizer output outside the chains. The recognize-only
// output itself fails the check (its chains keep alloc and copy), which is
// what `pre` produced before the copy was forwarded independently of the
// placement.
// CHECK-NOT: VIOLATION
// CHECK: LOAD CHAINS FORWARDED {{[1-9][0-9]*}}
// Q4_K loads %arg5 (the sub-block scales) three times. The restrict claim
// (Req 14.7 (3)) lets only one of them be forwarded; the other two chains
// keep their alloc, copy and writable to_tensor, unchanged from the
// recognizer output, and are the only violations.
// Q4K-COUNT-2: VIOLATION: private copy kept: ('memref.alloc'
// Q4K-NOT: VIOLATION: {{.*}}memref<192
// Q4K: LOAD CHAINS NOT FORWARDED {{[0-9]+}}
// RUN: not %python %S/Inputs/load_chain_check.py --forwarded %t.q80.ref < %t.q80.ref | FileCheck %s --check-prefix=OTF
// Discrimination: the same check on the on-the-fly output must fail, where
// each view carries an extra stride-0 dimension and each to_tensor the
// parallel dim with it (Req 7.9).
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict_noalias.ttshared.mlir | not %python %S/Inputs/load_chain_check.py --forwarded %t.q80.ref | FileCheck %s --check-prefix=OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir | not %python %S/Inputs/load_chain_check.py --forwarded %t.q40.ref | FileCheck %s --check-prefix=OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir | not %python %S/Inputs/load_chain_check.py --forwarded %t.q4k.ref | FileCheck %s --check-prefix=OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir | not %python %S/Inputs/load_chain_check.py --forwarded %t.iq2.ref | FileCheck %s --check-prefix=OTF
// `not` alone would also accept a crash, so the summary line the script
// prints only after a complete comparison is required.
// OTF: VIOLATION
// OTF: LOAD CHAINS NOT FORWARDED {{[0-9]+}}
