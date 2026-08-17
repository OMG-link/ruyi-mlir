// RUN: buddy-opt --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q80_q80_strict.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=Q80_STRICT
// RUN: buddy-opt --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q80_q80_partition.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=Q80_PARTITION
// RUN: buddy-opt --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q40_q80_canonical.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=Q40_Q80
// RUN: buddy-opt --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q4k_q8k_canonical.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=Q4K_Q8K
// RUN: buddy-opt --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=IQ2XXS_Q8K
// RUN: buddy-opt --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/synthetic_l3_canonical.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=SYNTHETIC_L3

// Q80_STRICT: [qgemm-recognize] candidates: 1
// Q80_STRICT: [qgemm-recognize] op 0 walk: ops=22 boundaries=13

// Q80_PARTITION: [qgemm-recognize] candidates: 1
// Q80_PARTITION: [qgemm-recognize] op 0 walk: ops=22 boundaries=13

// Q40_Q80: [qgemm-recognize] candidates: 1
// Q40_Q80: [qgemm-recognize] op 0 walk: ops=35 boundaries=16

// Q4K_Q8K: [qgemm-recognize] candidates: 1
// Q4K_Q8K: [qgemm-recognize] op 0 walk: ops=90 boundaries=31

// IQ2XXS_Q8K: [qgemm-recognize] candidates: 1
// IQ2XXS_Q8K: [qgemm-recognize] op 0 walk: ops=98 boundaries=37

// SYNTHETIC_L3: [qgemm-recognize] candidates: 1
// SYNTHETIC_L3: [qgemm-recognize] op 0 walk: ops=44 boundaries=23
