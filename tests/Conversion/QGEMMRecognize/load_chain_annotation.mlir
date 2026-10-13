// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q80_q80_strict.ttshared.mlir -o %t.q80
// RUN: FileCheck %s --check-prefix=Q80-CHAIN --input-file=%t.q80
// RUN: FileCheck %s --check-prefix=Q80-VIEW --input-file=%t.q80
// RUN: FileCheck %s --check-prefix=Q80-ALLOC --input-file=%t.q80
// RUN: FileCheck %s --check-prefix=Q80-COPY --input-file=%t.q80
// RUN: FileCheck %s --check-prefix=Q80-TT --input-file=%t.q80
// RUN: FileCheck %s --check-prefix=Q80-NEUTRAL --input-file=%t.q80
// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q40_q80_canonical.ttshared.mlir -o %t.q40
// RUN: FileCheck %s --check-prefix=Q40-TT --input-file=%t.q40
// RUN: FileCheck %s --check-prefix=Q40-COPY --input-file=%t.q40
// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/q4k_q8k_canonical.ttshared.mlir -o %t.q4k
// RUN: FileCheck %s --check-prefix=Q4K-TT --input-file=%t.q4k
// RUN: FileCheck %s --check-prefix=Q4K-COPY --input-file=%t.q4k
// RUN: buddy-opt --mlir-disable-threading --qgemm-simplify-dim-transforms --qgemm-recognize %S/Inputs/iq2xxs_q8k_canonical.ttshared.mlir -o %t.iq2
// RUN: FileCheck %s --check-prefix=IQ2-TT --input-file=%t.iq2
// RUN: FileCheck %s --check-prefix=IQ2-TABLE --input-file=%t.iq2
//
// Requirement 14.1 and 14.3, checked per capture (Req 12.17, 12.18).
//
// Each operand's memref source chain belongs to the operator with stage
// data_prep: the outermost memref view op, the memref.alloc, the memref.copy
// filling it and the bufferization.to_tensor reading it all carry qgemm.node
// with the root of the operator that consumes the load. The scalar and index
// plumbing of the same chain (arith.muli, arith.index_cast, memref.dim) and
// the constants stay root-neutral, because the offset index_cast is shared
// with the output store chain.
//
// Each capture has exactly one root, so the expected root id is 0. Each kind
// of chain op gets its own prefix, because the four ops of one chain are
// printed interleaved with the next chain's.

// One complete Q8_0 chain, in order, all four ops owned by root 0.
// Q80-CHAIN-LABEL: func.func @q80_q80_strict
// Q80-CHAIN: %[[VIEW:[a-z_0-9]+]] = memref.reinterpret_cast %{{[a-z0-9]+}} to offset: [0], sizes: [192, 512], strides: [512, 1] {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q80-CHAIN: %[[BUF:[a-z_0-9]+]] = memref.alloc() {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : memref<192x512xi8>
// Q80-CHAIN: memref.copy %[[VIEW]], %[[BUF]] {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q80-CHAIN: bufferization.to_tensor %[[BUF]] restrict writable {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : memref<192x512xi8> to tensor<192x512xi8>

// Four operand chains: four of each chain op, and no fifth.
// Q80-VIEW-COUNT-4: memref.reinterpret_cast %{{[a-z0-9]+}} to offset: {{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// Q80-VIEW-NOT: memref.reinterpret_cast %{{[a-z0-9]+}} to offset: {{.*}}qgemm.node
// Q80-ALLOC-COUNT-4: memref.alloc() {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q80-ALLOC-NOT: memref.alloc() {qgemm.node
// Q80-COPY-COUNT-4: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q80-COPY-NOT: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} {qgemm.node
// Q80-TT-COUNT-4: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q80-TT-NOT: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {qgemm.node

// Req 14.3: the scalar and index ops of a chain, and the constants they read,
// carry no qgemm.node. The store chain's view op is not a load chain and
// stays neutral as well.
// Q80-NEUTRAL-LABEL: func.func @q80_q80_strict
// Q80-NEUTRAL: %{{[a-z0-9_]+}} = arith.constant 32 : i32{{ *$}}
// Q80-NEUTRAL: %{{[a-z0-9]+}} = arith.muli %{{[a-z0-9]+}}, %{{[a-z0-9_]+}} : i32{{ *$}}
// Q80-NEUTRAL: %{{[a-z0-9]+}} = arith.index_cast %{{[a-z0-9]+}} : i32 to index{{ *$}}
// Q80-NEUTRAL: %{{[a-z0-9]+}} = arith.muli %{{[a-z0-9]+}}, %{{[a-z0-9_]+}} : index{{ *$}}
// Q80-NEUTRAL: memref.reinterpret_cast %{{[a-z0-9]+}} to offset: {{.*}} : memref<*xf32> to memref<192x32xf32, strided<[128, 1], offset: ?>>{{ *$}}
// Q80-NEUTRAL-NOT: arith.{{[a-z_]+}} %{{.*}}qgemm.node
// Q80-NEUTRAL-NOT: memref.dim {{.*}}qgemm.node

// Q4_0 x Q8_0: the same four chains, two of them with a dynamic offset.
// Q40-COPY-COUNT-4: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q40-COPY-NOT: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} {qgemm.node
// Q40-TT-COUNT-4: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q40-TT-NOT: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {qgemm.node

// Q4_K x Q8_K: eight operand chains, all owned by the one root.
// Q4K-COPY-COUNT-8: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q4K-COPY-NOT: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} {qgemm.node
// Q4K-TT-COUNT-8: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// Q4K-TT-NOT: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {qgemm.node

// iq2_xxs x Q8_K: five copied operand chains plus two lookup-table chains
// that have no alloc/copy pair, so their reinterpret_cast feeds
// bufferization.to_tensor directly. Both are data_prep; the memref.dim
// giving the table's extent is not.
// IQ2-TT-COUNT-7: bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
// IQ2-TT-NOT: bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}qgemm.node
// IQ2-TABLE-LABEL: func.func @iq2xxs_q8k_canonical
// IQ2-TABLE: memref.dim %{{[a-z0-9]+}}, %{{[a-z0-9_]+}} : memref<*xi8>{{ *$}}
// IQ2-TABLE: %[[TABLE:[a-z_0-9]+]] = memref.reinterpret_cast %{{[a-z0-9]+}} to offset: [0], sizes: [%{{[a-z0-9_]+}}], strides: [1] {qgemm.node = {root = 0 : i64, stage = "data_prep"}}
// IQ2-TABLE: bufferization.to_tensor %[[TABLE]] restrict {{.*}}qgemm.node = {root = 0 : i64, stage = "data_prep"}
