// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,PRE
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=PREG
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=OTFG
//
// Q4_0 x Q8_0 capture under both decompress values. The pass emits no remark.
// Value chain: the w_q nibble diamond (low/high branches joined by
// tensor.insert_slice, then linalg.transpose and tensor.collapse_shape).
// Factor_Chain (identified by hand): the a_d and w_d extf leaves.
// Under PRE the diamond stays in (n,k) space without m; under OTF the whole
// diamond is lifted into m as a unit and every op carries both m and n.
// The operand load chains are inside the operator, so under OTF the missing
// m is folded into the w_q view op at stride 0 and the reduction-dimension
// reshape is lifted with the diamond; under both modes the chain's private
// copy is dropped (Req 7.9, 14.7).

// Req 6.1: under PRE the computed placement equals the recognized input's
// placement and Separation finds no rewrite, so apart from the forwarded
// load copies the PRE output equals the recognize-only output.
// They do fail on the OTF output: the PREG-NOT lines below reject the shapes
// on-the-fly creates (the nibble diamond lifted into m).
// CHECK-NOT: remark
// CHECK-LABEL: func.func @q40_q80_canonical

// w_q load, pre mode: data_prep, n only, read from the view directly.
// PRE: %[[WQT:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}stage = "data_prep"{{.*}} : memref<32x256xi8, strided<[256, 1], offset: ?>> to tensor<32x256xi8>

// PRE mode: the leaf reshape, diamond join, transpose and collapse lack m.
// PRE: tensor.expand_shape %[[WQT]] {{.*}}stage = "data_prep"{{.*}} : tensor<32x256xi8> into tensor<32x16x16xi8>
// PRE: tensor.insert_slice {{.*}}stage = "data_prep"{{.*}} : tensor<32x16x16xi8> into tensor<32x16x16x2xi8>
// PRE: tensor.insert_slice {{.*}}stage = "data_prep"{{.*}} : tensor<32x16x16xi8> into tensor<32x16x16x2xi8>
// PRE: linalg.transpose ins(%{{[a-z0-9_]+}} : tensor<32x16x16x2xi8>) outs(%{{[a-z0-9_]+}} : tensor<32x16x2x16xi8>) {{.*}}stage = "data_prep"
// PRE: tensor.collapse_shape {{.*}}stage = "data_prep"{{.*}} : tensor<32x16x2x16xi8> into tensor<32x16x32xi8>

// OTF mode: m is the stride-0 dimension of the w_q view, so the load itself
// already carries both dims and feeds the reshape, which in turn feeds both
// nibble branches; the diamond carries m and n.
// OTF: %[[WQV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [192, 32, 256], strides: [0, 256, 1] {{.*}}stage = "compute"{{.*}} : memref<*xi8> to memref<192x32x256xi8, strided<[0, 256, 1], offset: ?>>
// OTF: %[[WQM:[a-z0-9_]+]] = bufferization.to_tensor %[[WQV]] restrict {{.*}}stage = "compute"{{.*}} : memref<192x32x256xi8, strided<[0, 256, 1], offset: ?>> to tensor<192x32x256xi8>
// OTF: %[[WQ:[a-z0-9_]+]] = tensor.expand_shape %[[WQM]] {{.*}}stage = "compute"{{.*}} : tensor<192x32x256xi8> into tensor<192x32x16x16xi8>
// OTF: linalg.generic {{.*}}ins(%[[WQ]], %{{[a-z0-9_]+}} : tensor<192x32x16x16xi8>, tensor<192x32x16x16xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16x16xi8>) {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}}ins(%[[WQ]], %{{[a-z0-9_]+}} : tensor<192x32x16x16xi8>, tensor<192x32x16x16xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16x16xi8>) {{.*}}stage = "compute"
// OTF: tensor.insert_slice {{.*}}stage = "compute"{{.*}} : tensor<192x32x16x16xi8> into tensor<192x32x16x16x2xi8>
// OTF: tensor.insert_slice {{.*}}stage = "compute"{{.*}} : tensor<192x32x16x16xi8> into tensor<192x32x16x16x2xi8>
// OTF: linalg.transpose ins(%{{[a-z0-9_]+}} : tensor<192x32x16x16x2xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16x2x16xi8>) permutation = [0, 1, 2, 4, 3] {{.*}}stage = "compute"
// OTF: tensor.collapse_shape {{.*}}stage = "compute"{{.*}} : tensor<192x32x16x2x16xi8> into tensor<192x32x16x32xi8>
// OTF: linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<192x32x16x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16x32xi16>) {{.*}}stage = "compute"

// Factor_Chain: a_d lacks n, w_d lacks m; each broadcast feeds the join.
// PRE: %[[AD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<192x16xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x16xf32>) {{.*}}stage = "data_prep"
// PRE: %[[ADN:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[AD]] : tensor<192x16xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16xf32>)
// PRE: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<32x16xf16>) outs(%{{[a-z0-9_]+}} : tensor<32x16xf32>) {{.*}}stage = "data_prep"
// PRE: %[[WDM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WD]] : tensor<32x16xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16xf32>)
// PRE: linalg.generic {{.*}}ins(%[[ADN]], %[[WDM]] : tensor<192x32x16xf32>, tensor<192x32x16xf32>)

// Under on-the-fly each factor chain is lifted at its own load: the missing
// dim is the stride-0 one of the view, so the extf already runs in (m,n,k)
// space and no broadcast is created.
// OTF: %[[ADN:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}stage = "compute"{{.*}} : memref<192x32x16xf16, strided<[16, 0, 1]>> to tensor<192x32x16xf16>
// OTF: %[[WDM:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}stage = "compute"{{.*}} : memref<192x32x16xf16, strided<[0, 16, 1], offset: ?>> to tensor<192x32x16xf16>
// OTF: %[[AD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[ADN]] : tensor<192x32x16xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16xf32>) {{.*}}stage = "compute"
// OTF: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WDM]] : tensor<192x32x16xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16xf32>) {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}}ins(%[[AD]], %[[WD]] : tensor<192x32x16xf32>, tensor<192x32x16xf32>)

// CHECK: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark
// Whole OTF output: no operand load keeps a private copy (Req 12.20).
// Pre drops the copies too (Req 14.7); the stride-0 OTF checks above are
// what tell the modes apart.
// OTFG-LABEL: func.func @q40_q80_canonical
// OTFG-NOT: memref.alloc
// OTFG-NOT: memref.copy
// OTFG: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// OTFG-NOT: memref.alloc
// OTFG-NOT: memref.copy
// Whole pre output: no shape that only on-the-fly creates.
// PREG-LABEL: func.func @q40_q80_canonical
// PREG-NOT: tensor<192x32x16x16xi8>
// PREG-NOT: tensor<192x32x16x16x2xi8>
// PREG-NOT: tensor<192x32x16x2x16xi8>
// PREG-NOT: tensor<192x32x16x32xi8>
// PREG: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// PREG-NOT: tensor<192x32x16x16xi8>
// PREG-NOT: tensor<192x32x16x16x2xi8>
// PREG-NOT: tensor<192x32x16x2x16xi8>
// PREG-NOT: tensor<192x32x16x32xi8>
