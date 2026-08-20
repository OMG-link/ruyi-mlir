// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q80_q80_strict.ttshared.mlir 2>&1 >/dev/null | FileCheck %s --check-prefix=Q80_STRICT
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q80_q80_partition.ttshared.mlir 2>&1 >/dev/null | FileCheck %s --check-prefix=Q80_PARTITION
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q40_q80_canonical.ttshared.mlir 2>&1 >/dev/null | FileCheck %s --check-prefix=Q40_Q80
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q4k_q8k_canonical.ttshared.mlir 2>&1 >/dev/null | FileCheck %s --check-prefix=Q4K_Q8K
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir 2>&1 >/dev/null | FileCheck %s --check-prefix=IQ2XXS_Q8K
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/synthetic_l3_canonical.ttshared.mlir 2>&1 >/dev/null | FileCheck %s --check-prefix=SYNTHETIC_L3

// The trace is read from stderr only; stdout is discarded by the redirection
// above.
// Q80_STRICT: [qgemm-recognize] candidates: 1
// Q80_STRICT: [qgemm-recognize] op 0 walk: ops=19 boundaries=13

// Q80_PARTITION: [qgemm-recognize] candidates: 1
// Q80_PARTITION: [qgemm-recognize] op 0 walk: ops=19 boundaries=13

// Q40_Q80: [qgemm-recognize] candidates: 1
// Q40_Q80: [qgemm-recognize] op 0 walk: ops=31 boundaries=16

// Q4K_Q8K: [qgemm-recognize] candidates: 1
// Q4K_Q8K: [qgemm-recognize] op 0 walk: ops=85 boundaries=30

// IQ2XXS_Q8K: [qgemm-recognize] candidates: 1
// IQ2XXS_Q8K: [qgemm-recognize] op 0 walk: ops=88 boundaries=33

// SYNTHETIC_L3: [qgemm-recognize] candidates: 1
// SYNTHETIC_L3: [qgemm-recognize] op 0 walk: ops=39 boundaries=23
