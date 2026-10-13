// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,NOCOPY,Q80
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q40_q80_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,NOCOPY,Q40
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q4k_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=Q4K
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/iq2xxs_q8k_canonical_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefixes=CHECK,NOCOPY,IQ2
//
// Req 12.20: under on-the-fly, each lifted operand load is a
// memref.reinterpret_cast carrying the parallel dimension it lacked at
// stride 0, its chain keeps no memref.alloc and no memref.copy, and the first
// arithmetic op of the chain carries both parallel dimensions.
//
// Discrimination: every check below fails on the pre output and on the
// recognize-only output, where each chain still copies into a private alloc
// and the stride-0 views and (m,n,k)-shaped first arithmetic ops do not exist.
// CHECK-NOT: remark

// No chain keeps a private copy. The lookup-table chains of iq2_xxs have none
// to begin with and are left untouched (Req 7.12), which the last run's
// positive checks on memref.dim confirm.
// NOCOPY-NOT: memref.alloc
// NOCOPY-NOT: memref.copy

// --- Q8_0 x Q8_0: m-carrying activation and n-carrying weight.
// Q80: %[[AV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [192, 32, 512], strides: [512, 0, 1] {{.*}} : memref<*xi8> to memref<192x32x512xi8, strided<[512, 0, 1]>>
// Q80: %[[AT:[a-z0-9_]+]] = bufferization.to_tensor %[[AV]] restrict {{.*}} : memref<192x32x512xi8, strided<[512, 0, 1]>> to tensor<192x32x512xi8>
// Q80: %[[WV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [192, 32, 512], strides: [0, 512, 1] {{.*}} : memref<*xi8> to memref<192x32x512xi8, strided<[0, 512, 1], offset: ?>>
// Q80: %[[WT:[a-z0-9_]+]] = bufferization.to_tensor %[[WV]] restrict {{.*}} : memref<192x32x512xi8, strided<[0, 512, 1], offset: ?>> to tensor<192x32x512xi8>
// The first arithmetic op of the activation chain runs in (m,n,k) space.
// Q80: %[[AE:[a-z0-9_]+]] = tensor.expand_shape %[[AT]] {{.*}} : tensor<192x32x512xi8> into tensor<192x32x16x32xi8>
// Q80: linalg.generic {{.*}}ins(%[[AE]] : tensor<192x32x16x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16x32xi16>) {{.*}}stage = "compute"
// Q80: arith.extsi

// --- Q4_0 x Q8_0: the weight nibble chain gains m at stride 0.
// Q40: %[[WV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [192, 32, 256], strides: [0, 256, 1] {{.*}} : memref<*xi8> to memref<192x32x256xi8, strided<[0, 256, 1], offset: ?>>
// Q40: %[[WT:[a-z0-9_]+]] = bufferization.to_tensor %[[WV]] restrict {{.*}} : memref<192x32x256xi8, strided<[0, 256, 1], offset: ?>> to tensor<192x32x256xi8>
// Q40: %[[WE:[a-z0-9_]+]] = tensor.expand_shape %[[WT]] {{.*}} : tensor<192x32x256xi8> into tensor<192x32x16x16xi8>
// Q40: linalg.generic {{.*}}ins(%[[WE]], %{{[a-z0-9_]+}} : tensor<192x32x16x16xi8>, tensor<192x32x16x16xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x32x16x16xi8>) {{.*}}stage = "compute"
// Q40: arith.andi

// --- Q4_K x Q8_K: the weight qs chain gains m at stride 0.
// Q4K: %[[WV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [192, 8, 32, 32], strides: [0, 4096, 128, 1] {{.*}} : memref<*xi8> to memref<192x8x32x32xi8, strided<[0, 4096, 128, 1], offset: ?>>
// Q4K: %[[WT:[a-z0-9_]+]] = bufferization.to_tensor %[[WV]] restrict {{.*}} : memref<192x8x32x32xi8, strided<[0, 4096, 128, 1], offset: ?>> to tensor<192x8x32x32xi8>
// Q4K: linalg.generic {{.*}}ins(%[[WT]], %{{[a-z0-9_]+}} : tensor<192x8x32x32xi8>, tensor<192x8x32x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<192x8x32x32xi8>) {{.*}}stage = "compute"
// Q4K: arith.andi

// --- iq2_xxs x Q8_K: the qs chain gains m at stride 0; the grid and sign
// lookup chains keep their copy-free recognized form.
// IQ2: %[[QV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [192, 2, 32, 32], strides: [0, 4096, 128, 1] {{.*}} : memref<*xi16> to memref<192x2x32x32xi16, strided<[0, 4096, 128, 1], offset: ?>>
// IQ2: %[[QT:[a-z0-9_]+]] = bufferization.to_tensor %[[QV]] restrict {{.*}} : memref<192x2x32x32xi16, strided<[0, 4096, 128, 1], offset: ?>> to tensor<192x2x32x32xi16>
// IQ2: %[[QE:[a-z0-9_]+]] = tensor.expand_shape %[[QT]] {{.*}} : tensor<192x2x32x32xi16> into tensor<192x2x16x2x32xi16>
// IQ2: linalg.generic {{.*}}ins(%[[QE]] : tensor<192x2x16x2x32xi16>) outs(%{{[a-z0-9_]+}} : tensor<192x2x16x2x32xi32>) {{.*}}stage = "compute"
// IQ2: arith.extui
// The lookup-table chain is untouched: memref.dim, a dynamic view and a
// to_tensor read through tensor.extract, with no parallel dimension folded in.
// IQ2: memref.dim
// IQ2: %[[GV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [%{{[a-z0-9_]+}}], strides: [1] {{.*}}stage = "data_prep"{{.*}} : memref<*xi8> to memref<?xi8>
// IQ2: bufferization.to_tensor %[[GV]] restrict {{.*}}stage = "data_prep"{{.*}} : memref<?xi8> to tensor<?xi8>

// NOCOPY-NOT: memref.alloc
// NOCOPY-NOT: memref.copy
// CHECK-NOT: remark
