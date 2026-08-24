// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q80_q80_strict.ttshared.mlir > %t.q80_strict.ir 2> %t.q80_strict.trace
// RUN: FileCheck %s --check-prefix=Q80_STRICT_IR < %t.q80_strict.ir
// RUN: FileCheck %s --check-prefix=Q80_STRICT_ROOT < %t.q80_strict.ir
// RUN: FileCheck %s --check-prefix=Q80_STRICT_TRACE < %t.q80_strict.trace
// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q80_q80_partition.ttshared.mlir > %t.q80_partition.ir 2> %t.q80_partition.trace
// RUN: FileCheck %s --check-prefix=Q80_PARTITION_IR < %t.q80_partition.ir
// RUN: FileCheck %s --check-prefix=Q80_PARTITION_ROOT < %t.q80_partition.ir
// RUN: FileCheck %s --check-prefix=Q80_PARTITION_TRACE < %t.q80_partition.trace
// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q40_q80_canonical.ttshared.mlir > %t.q40_q80.ir 2> %t.q40_q80.trace
// RUN: FileCheck %s --check-prefix=Q40_Q80_IR < %t.q40_q80.ir
// RUN: FileCheck %s --check-prefix=Q40_Q80_ROOT < %t.q40_q80.ir
// RUN: FileCheck %s --check-prefix=Q40_Q80_TRACE < %t.q40_q80.trace
// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q4k_q8k_canonical.ttshared.mlir > %t.q4k_q8k.ir 2> %t.q4k_q8k.trace
// RUN: FileCheck %s --check-prefix=Q4K_Q8K_IR < %t.q4k_q8k.ir
// RUN: FileCheck %s --check-prefix=Q4K_Q8K_ROOT < %t.q4k_q8k.ir
// RUN: FileCheck %s --check-prefix=Q4K_Q8K_TRACE < %t.q4k_q8k.trace
// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir > %t.iq2xxs_q8k.ir 2> %t.iq2xxs_q8k.trace
// RUN: FileCheck %s --check-prefix=IQ2XXS_Q8K_IR < %t.iq2xxs_q8k.ir
// RUN: FileCheck %s --check-prefix=IQ2XXS_Q8K_ROOT < %t.iq2xxs_q8k.ir
// RUN: FileCheck %s --check-prefix=IQ2XXS_Q8K_TRACE < %t.iq2xxs_q8k.trace
// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/synthetic_l3_canonical.ttshared.mlir > %t.synthetic_l3.ir 2> %t.synthetic_l3.trace
// RUN: FileCheck %s --check-prefix=SYNTHETIC_L3_IR < %t.synthetic_l3.ir
// RUN: FileCheck %s --check-prefix=SYNTHETIC_L3_ROOT < %t.synthetic_l3.ir
// RUN: FileCheck %s --check-prefix=SYNTHETIC_L3_TRACE < %t.synthetic_l3.trace

// Every capture checks both the annotated stdout and the debug trace.  The
// stage checks use operation kinds rather than SSA names; the root checks
// prove that only the rank-2 final reduction carries qgemm.root.

// Q80_STRICT_IR-LABEL: func.func @q80_q80_strict
// Q80_STRICT_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// Q80_STRICT_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// Q80_STRICT_IR: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// Q80_STRICT_IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// Q80_STRICT_ROOT-LABEL: func.func @q80_q80_strict
// Q80_STRICT_ROOT-NOT: qgemm.root
// Q80_STRICT_ROOT: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// Q80_STRICT_ROOT-NOT: qgemm.root
// Q80_STRICT_TRACE: [qgemm-recognize] candidates: 1
// Q80_STRICT_TRACE: [qgemm-recognize] op 0 walk: ops=17 boundaries=13
// Q80_STRICT_TRACE: [qgemm-recognize] op 0 accepted: root=0

// Q80_PARTITION_IR-LABEL: func.func @q80_q80_partition_m
// Q80_PARTITION_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// Q80_PARTITION_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// Q80_PARTITION_IR: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// Q80_PARTITION_IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// Q80_PARTITION_ROOT-LABEL: func.func @q80_q80_partition_m
// Q80_PARTITION_ROOT-NOT: qgemm.root
// Q80_PARTITION_ROOT: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// Q80_PARTITION_ROOT-NOT: qgemm.root
// Q80_PARTITION_TRACE: [qgemm-recognize] candidates: 1
// Q80_PARTITION_TRACE: [qgemm-recognize] op 0 walk: ops=17 boundaries=13
// Q80_PARTITION_TRACE: [qgemm-recognize] op 0 accepted: root=0

// Q40_Q80_IR-LABEL: func.func @q40_q80_canonical
// Q40_Q80_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// Q40_Q80_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// Q40_Q80_IR: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// Q40_Q80_IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// Q40_Q80_ROOT-LABEL: func.func @q40_q80_canonical
// Q40_Q80_ROOT-NOT: qgemm.root
// Q40_Q80_ROOT: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// Q40_Q80_ROOT-NOT: qgemm.root
// Q40_Q80_TRACE: [qgemm-recognize] candidates: 1
// Q40_Q80_TRACE: [qgemm-recognize] op 0 walk: ops=26 boundaries=18
// Q40_Q80_TRACE: [qgemm-recognize] op 0 accepted: root=0

// Q4K_Q8K_IR-LABEL: func.func @q4k_q8k_canonical
// Q4K_Q8K_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// Q4K_Q8K_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// Q4K_Q8K_IR: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// Q4K_Q8K_IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// Q4K_Q8K_ROOT-LABEL: func.func @q4k_q8k_canonical
// Q4K_Q8K_ROOT-NOT: qgemm.root
// Q4K_Q8K_ROOT: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// Q4K_Q8K_ROOT-NOT: qgemm.root
// Q4K_Q8K_TRACE: [qgemm-recognize] candidates: 1
// Q4K_Q8K_TRACE: [qgemm-recognize] op 0 walk: ops=76 boundaries=36
// Q4K_Q8K_TRACE: [qgemm-recognize] op 0 accepted: root=0

// IQ2XXS_Q8K_IR-LABEL: func.func @iq2xxs_q8k_canonical
// IQ2XXS_Q8K_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// IQ2XXS_Q8K_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// IQ2XXS_Q8K_IR: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// IQ2XXS_Q8K_IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// IQ2XXS_Q8K_ROOT-LABEL: func.func @iq2xxs_q8k_canonical
// IQ2XXS_Q8K_ROOT-NOT: qgemm.root
// IQ2XXS_Q8K_ROOT: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// IQ2XXS_Q8K_ROOT-NOT: qgemm.root
// IQ2XXS_Q8K_TRACE: [qgemm-recognize] candidates: 1
// IQ2XXS_Q8K_TRACE: [qgemm-recognize] op 0 walk: ops=80 boundaries=44
// IQ2XXS_Q8K_TRACE: [qgemm-recognize] op 0 accepted: root=0

// SYNTHETIC_L3_IR-LABEL: func.func @synthetic_l3_canonical
// SYNTHETIC_L3_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// SYNTHETIC_L3_IR-DAG: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// SYNTHETIC_L3_IR: linalg.reduce{{.*}}dimensions = [3]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// SYNTHETIC_L3_IR: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// SYNTHETIC_L3_ROOT-LABEL: func.func @synthetic_l3_canonical
// SYNTHETIC_L3_ROOT-NOT: qgemm.root
// SYNTHETIC_L3_ROOT: linalg.reduce{{.*}}dimensions = [2]{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// SYNTHETIC_L3_ROOT-NOT: qgemm.root
// SYNTHETIC_L3_TRACE: [qgemm-recognize] candidates: 1
// SYNTHETIC_L3_TRACE: [qgemm-recognize] op 0 walk: ops=38 boundaries=26
// SYNTHETIC_L3_TRACE: [qgemm-recognize] op 0 accepted: root=0
