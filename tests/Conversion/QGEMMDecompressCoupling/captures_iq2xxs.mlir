// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,PRE
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=PREG
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=OTFG
//
// iq2_xxs x Q8_K capture under both decompress values. The pass emits no
// remark. The weight qs words feed a shared prefix (extsi, transpose,
// extract_slice) used by both the value chain (grid lookup through
// tensor.extract / linalg.index) and the Factor_Chain (identified by hand:
// the w_d leaf, its extf, the product with the qs-derived sub-block scale
// and the final transpose). The qs load chain belongs to the operator.
// Its private copy is dropped under both modes (Req 14.7); under on-the-fly m
// is folded into its view op at stride 0 and the
// expand_shape leaf is lifted with the prefix (Req 7.9, 14.7). The grid and
// sign lookup-table chains, read through tensor.extract inside a region
// without a copy, are untouched in both modes (Req 7.12).
// Under PRE no op from the shared prefix through the end of the
// Factor_Chain carries m, and the Factor_Chain's m broadcast feeds the join
// directly. Under OTF the prefix, value chain and Factor_Chain are lifted
// into m and n.

// The PRE checks also hold facts the recognize-only output violates: the
// k-reduction's zero init is built as a 2x8 expand_shape without m or n and
// reaches the reduction through the m and then the n broadcast, and no
// 192x32x16 init (the recognized, broadcast-first form) remains.
// CHECK-NOT: remark
// CHECK-LABEL: func.func @iq2xxs_q8k_canonical

// qs load, pre mode: n only, read from the view directly.
// PRE: %[[QS:[a-z0-9_]+]] = bufferization.to_tensor {{.*}} restrict {{.*}} : memref<2x32x32xi16, strided<[4096, 128, 1], offset: ?>> to tensor<2x32x32xi16>


// PRE mode: the qs leaf keeps its recognized shape, data_prep, n only.
// PRE: %[[QSE:[a-z0-9_]+]] = tensor.expand_shape %[[QS]] {{.*}}stage = "data_prep"{{.*}} : tensor<2x32x32xi16> into tensor<2x16x2x32xi16>

// PRE mode: the shared prefix has no m axis.
// PRE: linalg.generic {{.*}}ins(%[[QSE]] : tensor<2x16x2x32xi16>) outs(%{{[a-z0-9_]+}} : tensor<2x16x2x32xi32>) {{.*}}stage = "data_prep"
// PRE: linalg.transpose ins(%{{[a-z0-9_]+}} : tensor<2x16x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<2x16x32x2xi32>) permutation = [0, 1, 3, 2] {{.*}}stage = "data_prep"
// PRE: tensor.extract_slice {{.*}}stage = "data_prep"{{.*}} : tensor<2x16x32x2xi32> to tensor<2x16x32xi32>
// PRE-NOT: tensor<192x
// PRE: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<2x32xf16>) outs(%{{[a-z0-9_]+}} : tensor<2x8x32xf16>) {{.*}}stage = "data_prep"
// PRE-NOT: tensor<192x
// PRE: linalg.generic {{.*}}ins(%[[WD]] : tensor<2x8x32xf16>) outs(%{{[a-z0-9_]+}} : tensor<2x8x32xf32>) {{.*}}stage = "data_prep"
// PRE-NOT: tensor<192x
// PRE: %[[FT:[a-z0-9_]+]] = linalg.transpose ins(%{{[a-z0-9_]+}} : tensor<2x8x32xf32>) outs(%{{[a-z0-9_]+}} : tensor<32x2x8xf32>) permutation = [2, 0, 1] {{.*}}stage = "data_prep"
// PRE mode: the Factor_Chain's m broadcast feeds the join.
// PRE: %[[FTM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[FT]] : tensor<32x2x8xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8xf32>) {{.*}}stage = "compute"
// PRE mode: the k-reduction's init is built without m or n, then broadcast
// along m and n; the recognize-only output builds it as 192x32x16.
// PRE: %[[INIT:[a-z0-9_]+]] = tensor.expand_shape {{.*}}stage = "data_prep"{{.*}} : tensor<16xi32> into tensor<2x8xi32>
// PRE: %[[INITM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[INIT]] : tensor<2x8xi32>) outs(%{{[a-z0-9_]+}} : tensor<192x2x8xi32>) {{.*}}stage = "compute"
// PRE: %[[INITMN:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[INITM]] : tensor<192x2x8xi32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8xi32>) {{.*}}stage = "compute"
// PRE: linalg.generic {{.*}}"reduction"]} ins(%{{[a-z0-9_]+}} : tensor<192x32x2x8x32xi32>) outs(%[[INITMN]] : tensor<192x32x2x8xi32>) {{.*}}stage = "compute"
// PRE: linalg.generic {{.*}}ins(%{{[a-z0-9_]+}}, %[[FTM]] : tensor<192x32x2x8xf32>, tensor<192x32x2x8xf32>)

// OTF mode: m is the stride-0 dimension of the qs view, so the load itself
// carries it, and the leaf expand_shape and the whole shared prefix are
// lifted into m.
// OTF: %[[QSV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [192, 2, 32, 32], strides: [0, 4096, 128, 1] {{.*}}stage = "compute"{{.*}} : memref<*xi16> to memref<192x2x32x32xi16, strided<[0, 4096, 128, 1], offset: ?>>
// OTF: %[[QSM:[a-z0-9_]+]] = bufferization.to_tensor %[[QSV]] restrict {{.*}}stage = "compute"{{.*}} : memref<192x2x32x32xi16, strided<[0, 4096, 128, 1], offset: ?>> to tensor<192x2x32x32xi16>
// The w_d view is lifted the same way, right next to the qs one.
// OTF: %[[WDV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [192, 2, 32], strides: [0, 128, 1] {{.*}}stage = "compute"{{.*}} : memref<*xf16> to memref<192x2x32xf16, strided<[0, 128, 1], offset: ?>>
// OTF: %[[WDT0:[a-z0-9_]+]] = bufferization.to_tensor %[[WDV]] restrict {{.*}}stage = "compute"{{.*}} : memref<192x2x32xf16, strided<[0, 128, 1], offset: ?>> to tensor<192x2x32xf16>
// OTF: %[[QSE:[a-z0-9_]+]] = tensor.expand_shape %[[QSM]] {{.*}}stage = "compute"{{.*}} : tensor<192x2x32x32xi16> into tensor<192x2x16x2x32xi16>
// OTF: linalg.generic {{.*}}ins(%[[QSE]] : tensor<192x2x16x2x32xi16>) outs(%{{[a-z0-9_]+}} : tensor<192x2x16x2x32xi32>) {{.*}}stage = "compute"
// OTF: linalg.transpose ins(%{{[a-z0-9_]+}} : tensor<192x2x16x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<192x2x16x32x2xi32>) permutation = [0, 1, 2, 4, 3] {{.*}}stage = "compute"
// OTF: tensor.extract_slice {{.*}}stage = "compute"{{.*}} : tensor<192x2x16x32x2xi32> to tensor<192x2x16x32xi32>
// OTF mode: an index-only lookup generic carries m and n.
// OTF: linalg.generic {{.*}}outs(%{{[a-z0-9_]+}} : tensor<192x4x32xi32>) {{.*}}stage = "compute"
// OTF: linalg.index
// OTF mode: value-chain grid lookup carries m and n.
// OTF: linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<192x16x4x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<192x16x4x32x8xi8>) {{.*}}stage = "compute"
// OTF: linalg.index
// OTF: tensor.extract
// OTF mode: Factor_Chain: m is the stride-0 dimension of the w_d view, so
// the leaf sub-block broadcast and the extf are both in H_m.
// OTF: %[[WDM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WDT0]] : tensor<192x2x32xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x2x8x32xf16>) {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}}ins(%[[WDM]] : tensor<192x2x8x32xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x2x8x32xf32>) {{.*}}stage = "compute"
// OTF: linalg.transpose ins(%{{[a-z0-9_]+}} : tensor<192x2x8x32xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8xf32>) permutation = [0, 3, 1, 2] {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<192x32x2x8x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8x32xi16>) {{.*}}stage = "compute"

// CHECK: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark
// Whole on-the-fly output: no operand load keeps a private copy (Req 12.20).
// Pre drops the copies too (Req 14.7); the stride-0 OTF checks above are
// what tell the modes apart.
// OTFG-LABEL: func.func @iq2xxs_q8k_canonical
// OTFG-NOT: memref.alloc
// OTFG-NOT: memref.copy
// OTFG: memref.dim
// OTFG: memref.reinterpret_cast
// OTFG: bufferization.to_tensor
// OTFG-NOT: memref.alloc
// OTFG-NOT: memref.copy
// Whole pre output: no shape that only on-the-fly creates.
// PREG-LABEL: func.func @iq2xxs_q8k_canonical
// PREG-NOT: tensor<192x32x16xi32>
// PREG-NOT: tensor<192x2x16x2x32xi16>
// PREG-NOT: tensor<192x16x4x32x8xi8>
// PREG: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// PREG-NOT: tensor<192x32x16xi32>
// PREG-NOT: tensor<192x2x16x2x32xi16>
// PREG-NOT: tensor<192x16x4x32x8xi8>
