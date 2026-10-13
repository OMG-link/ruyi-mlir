// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,PRE
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=OTFG
//
// Q4_K x Q8_K capture under both decompress values. The pass emits no remark.
// Zero-point degenerate path: the a_q leaf is sign-extended, block-summed
// over the sub-block, converted to f32 and multiplied by the n-varying dmin
// factor. Under PRE the block sum has no n axis and its n broadcast feeds the
// join with the dmin factor directly; under OTF the block sum is in L_n and
// the n broadcast takes the a_q load itself.
// Factor_Chain (identified by hand): the w_d super-block scale (transpose
// leaf, extf), joined with the a_d scale.
// The operand load chains are part of the operator, so the Boundary_Load of
// each operand is its outermost memref view op; the bufferization.to_tensor
// and the first tensor op after it are interior nodes and become eligible
// for H_d. The chain's private copy is dropped under both modes (Req 14.7); under
// on-the-fly the missing parallel dim is also folded into the view op at
// stride 0 (Req 7.9). Exception: %arg5 (sub-block scales) is loaded three
// times, so the restrict claim (Req 14.7 (3)) forwards one of them; under OTF
// the two kept copies get R3 on view and copy (Req 14.8).
// PRE-NOT: remark
// OTF-COUNT-4: remark: qgemm-decompress-coupling: 'm' cannot be inserted into op
// OTF-NOT: remark
// CHECK-LABEL: func.func @q4k_q8k_canonical

// a_q load, pre mode: data_prep, m only, read from the view directly.
// PRE: %[[AQT:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}stage = "data_prep"{{.*}} : memref<192x512xi8, strided<[512, 1]>> to tensor<192x512xi8>

// PRE mode: the leaf reshape and the value-chain extension lack n.
// PRE: %[[AQ:[a-z0-9_]+]] = tensor.expand_shape %[[AQT]] {{.*}}stage = "data_prep"{{.*}} : tensor<192x512xi8> into tensor<192x2x8x32xi8>
// PRE mode: zero-point path: block sum without n, n broadcast feeds the dmin join.
// PRE: %[[ZX:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[AQ]] : tensor<192x2x8x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x2x8x32xi32>) {{.*}}stage = "data_prep"
// PRE: %[[BS:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[ZX]] : tensor<192x2x8x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<192x2x8xi32>) {{.*}}stage = "data_prep"
// PRE: %[[BSF:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[BS]] : tensor<192x2x8xi32>) outs(%{{[a-z0-9_]+}} : tensor<192x2x8xf32>) {{.*}}stage = "data_prep"
// PRE: %[[BSN:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[BSF]] : tensor<192x2x8xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8xf32>) {{.*}}stage = "compute"
// PRE: %[[DMN:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<32x2x8xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8xf32>) {{.*}}stage = "compute"
// PRE: linalg.generic {{.*}}ins(%[[BSN]], %[[DMN]] : tensor<192x32x2x8xf32>, tensor<192x32x2x8xf32>)

// OTF mode: n is the stride-0 dimension of the a_q view, so the load itself
// carries it and the leaf reshape, the value-chain extension (H_n) and the
// zero-point block sum all carry n.
// OTF: %[[AQV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [192, 32, 512], strides: [512, 0, 1] {{.*}}stage = "compute"{{.*}} : memref<*xi8> to memref<192x32x512xi8, strided<[512, 0, 1]>>
// OTF: %[[AQN:[a-z0-9_]+]] = bufferization.to_tensor %[[AQV]] restrict {{.*}}stage = "compute"{{.*}} : memref<192x32x512xi8, strided<[512, 0, 1]>> to tensor<192x32x512xi8>
// OTF: %[[AQ:[a-z0-9_]+]] = tensor.expand_shape %[[AQN]] {{.*}}stage = "compute"{{.*}} : tensor<192x32x512xi8> into tensor<192x32x2x8x32xi8>
// OTF: linalg.generic {{.*}}ins(%[[AQ]] : tensor<192x32x2x8x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8x32xi16>) {{.*}}stage = "compute"
// OTF: %[[ZX:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[AQ]] : tensor<192x32x2x8x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8x32xi32>) {{.*}}stage = "compute"
// OTF: %[[BS:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[ZX]] : tensor<192x32x2x8x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8xi32>) {{.*}}stage = "compute"
// OTF: %[[BSF:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[BS]] : tensor<192x32x2x8xi32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2x8xf32>) {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}}ins(%[[BSF]], %{{[a-z0-9_]+}} : tensor<192x32x2x8xf32>, tensor<192x32x2x8xf32>)

// Factor_Chain: w_d scale. Under pre its load is data_prep with n only.
// PRE: %[[WDL:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}stage = "data_prep"{{.*}} : memref<2x32xf16, strided<[128, 1], offset: ?>> to tensor<2x32xf16>

// PRE mode: the transpose and the extf lack m; the m broadcast of the extf
// result feeds the join with the a_d scale.
// PRE: %[[WDT:[a-z0-9_]+]] = linalg.transpose ins(%[[WDL]] : tensor<2x32xf16>) outs(%{{[a-z0-9_]+}} : tensor<32x2xf16>) permutation = [1, 0] {{.*}}stage = "data_prep"
// PRE: %[[WDF:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WDT]] : tensor<32x2xf16>) outs(%{{[a-z0-9_]+}} : tensor<32x2xf32>) {{.*}}stage = "data_prep"
// PRE: %[[WDM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WDF]] : tensor<32x2xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2xf32>) {{.*}}stage = "compute"
// PRE: %[[ADF:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<192x2xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x2xf32>) {{.*}}stage = "data_prep"
// PRE: %[[ADN:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[ADF]] : tensor<192x2xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2xf32>) {{.*}}stage = "compute"
// PRE: linalg.generic {{.*}}ins(%[[WDM]], %[[ADN]] : tensor<192x32x2xf32>, tensor<192x32x2xf32>)

// OTF mode: m is the stride-0 dimension of the w_d view, so the transpose and
// the extf are both in H_m and read the load directly.
// OTF: %[[WDV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [192, 2, 32], strides: [0, 128, 1] {{.*}}stage = "compute"{{.*}} : memref<*xf16> to memref<192x2x32xf16, strided<[0, 128, 1], offset: ?>>
// OTF: %[[WDM:[a-z0-9_]+]] = bufferization.to_tensor %[[WDV]] restrict {{.*}}stage = "compute"{{.*}} : memref<192x2x32xf16, strided<[0, 128, 1], offset: ?>> to tensor<192x2x32xf16>
// OTF: %[[WDT:[a-z0-9_]+]] = linalg.transpose ins(%[[WDM]] : tensor<192x2x32xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2xf16>) permutation = [0, 2, 1] {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}}ins(%[[WDT]] : tensor<192x32x2xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x32x2xf32>) {{.*}}stage = "compute"

// CHECK: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark

// Whole on-the-fly output: no operand load keeps a private copy (Req 12.20)
// except the two %arg5 scale loads kept by the restrict claim.
// Pre drops the copies too (Req 14.7); the stride-0 OTF checks above are
// what tell the modes apart.
// OTFG-LABEL: func.func @q4k_q8k_canonical
// OTFG-NOT: memref.copy
// OTFG: memref.copy {{.*}}to memref<2x4x32xi8>
// OTFG-NOT: memref.copy
// OTFG: memref.copy {{.*}}to memref<2x4x32xi8>
// OTFG-NOT: memref.alloc
// OTFG-NOT: memref.copy
// OTFG: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// OTFG-NOT: memref.alloc
// OTFG-NOT: memref.copy
