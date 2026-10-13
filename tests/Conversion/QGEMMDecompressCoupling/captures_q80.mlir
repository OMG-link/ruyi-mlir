// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q80_q80_strict_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,PRE
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q80_q80_strict_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=PREG
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=OTFG
//
// Q8_0 x Q8_0 capture under both decompress values. The pass emits no remark.
// Value chain: the a_q leaf and its sign extension. Factor_Chain (identified
// by hand): the a_d and w_d extf leaves, joined by the scale product.
// The operand load chains belong to the operator now, so the Boundary_Load of
// each operand is the outermost memref view op and the bufferization.to_tensor
// is an interior node. Under both modes the chain's private copy is dropped
// (Req 14.7). Under PRE the chain stays out of the other parallel dim. Under
// OTF the missing parallel dim is folded into the view op at stride 0, so no broadcast is left on the load edge at all
// and the reduction-dimension reshape and the first arithmetic op of each
// chain are lifted too (Req 7.9, 7.10, 14.7).

// Req 6.1: under PRE the computed placement equals the recognized input's
// placement (decompression already stays out of the other parallel dim) and
// Separation finds no rewrite, so the PRE output equals the recognize-only
// output; the PRE checks cannot and do not tell the two apart. They do fail
// on the OTF output: the PREG-NOT lines below reject the shapes on-the-fly
// creates (the n-extended raw a_q load).
// CHECK-NOT: remark
// CHECK-LABEL: func.func @q80_q80_strict

// Value chain load, pre mode: data_prep, m only, the to_tensor reads the
// view directly with no private copy (Req 14.7) and is no longer writable.
// PRE: %[[AQT:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}stage = "data_prep"{{.*}} : memref<192x512xi8, strided<[512, 1]>> to tensor<192x512xi8>

// PRE mode: the reshape and the extension lack n, the n broadcast follows.
// PRE: %[[AQ:[a-z0-9_]+]] = tensor.expand_shape %[[AQT]] {{.*}}stage = "data_prep"{{.*}} : tensor<192x512xi8> into tensor<192x16x32xi8>
// PRE: %[[AQX:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[AQ]] : tensor<192x16x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x16x32xi16>) {{.*}}stage = "data_prep"
// PRE: linalg.generic {{.*}}ins(%[[AQX]] : tensor<192x16x32xi16>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16x32xi16>) {{.*}}stage = "compute"

// OTF mode: n is a stride-0 dimension of the view, the alloc and the copy are
// gone, and the reshape and the extension are both in H_n.
// The view takes n where the multiply reads it, right after m.
// OTF: %[[AQV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [192, 32, 512], strides: [512, 0, 1] {{.*}}stage = "compute"{{.*}} : memref<*xi8> to memref<192x32x512xi8, strided<[512, 0, 1]>>
// OTF: %[[AQT:[a-z0-9_]+]] = bufferization.to_tensor %[[AQV]] restrict {{.*}}stage = "compute"{{.*}} : memref<192x32x512xi8, strided<[512, 0, 1]>> to tensor<192x32x512xi8>
// OTF: %[[AQNE:[a-z0-9_]+]] = tensor.expand_shape %[[AQT]] {{.*}}stage = "compute"{{.*}} : tensor<192x32x512xi8> into tensor<192x32x16x32xi8>
// OTF: linalg.generic {{.*}}ins(%[[AQNE]] : tensor<192x32x16x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16x32xi16>) {{.*}}stage = "compute"

// Factor_Chain: a_d lacks n, w_d lacks m; each broadcast feeds the join.
// PRE: %[[AD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<192x16xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x16xf32>) {{.*}}stage = "data_prep"
// PRE: %[[ADN:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[AD]] : tensor<192x16xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16xf32>)
// PRE: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<32x16xf16>) outs(%{{[a-z0-9_]+}} : tensor<32x16xf32>) {{.*}}stage = "data_prep"
// PRE: %[[WDM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WD]] : tensor<32x16xf32>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16xf32>)
// PRE: linalg.generic {{.*}}ins(%[[ADN]], %[[WDM]] : tensor<192x32x16xf32>, tensor<192x32x16xf32>)

// Under on-the-fly each factor chain is lifted at its own load: the missing
// dim is the stride-0 one of the view, so the extf already runs in (m,n,k)
// space and no broadcast is created.
// OTF: %[[ADT:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}stage = "compute"{{.*}} : memref<192x32x16xf16, strided<[16, 0, 1]>> to tensor<192x32x16xf16>
// OTF: %[[WDT:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict {{.*}}stage = "compute"{{.*}} : memref<192x32x16xf16, strided<[0, 16, 1], offset: ?>> to tensor<192x32x16xf16>
// OTF: %[[AD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[ADT]] : tensor<192x32x16xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16xf32>) {{.*}}stage = "compute"
// OTF: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WDT]] : tensor<192x32x16xf16>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16xf32>) {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}}ins(%[[AD]], %[[WD]] : tensor<192x32x16xf32>, tensor<192x32x16xf32>)

// CHECK: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark
// Whole OTF output: no operand load keeps a private copy (Req 12.20).
// Pre drops the copies too (Req 14.7); the stride-0 OTF checks above are
// what tell the modes apart.
// OTFG-LABEL: func.func @q80_q80_strict
// OTFG-NOT: memref.alloc
// OTFG-NOT: memref.copy
// OTFG: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// OTFG-NOT: memref.alloc
// OTFG-NOT: memref.copy
// Whole PRE output: no shape that only on-the-fly creates.
// PREG-LABEL: func.func @q80_q80_strict
// PREG-NOT: tensor<192x32x16x32xi8>
// PREG: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// PREG-NOT: tensor<192x32x16x32xi8>
