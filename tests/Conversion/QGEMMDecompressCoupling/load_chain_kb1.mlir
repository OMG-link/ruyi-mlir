// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_kb1_noalias.ttshared.mlir 2>&1 | FileCheck %s
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_kb1_noalias.ttshared.mlir 2>&1 | FileCheck %s --check-prefix=NOCOPY
//
// Req 12.24 (with Req 6.8, 7.4): Q8_0 x Q8_0 at M=64, N=32, K=32, so there is
// one quantization block along K (KB=1). --qgemm-simplify-dim-transforms
// drops the unit kb dim of the a_d chain with a tensor.collapse_shape
// [[0, 1]] 64x1 -> 64 that groups m with kb. Under on-the-fly the a_d chain
// is lifted into n. Its consumer reads n right after m, which in the
// collapse_shape's source is right after the whole m/kb group: the
// collapse_shape gains a new singleton group for n behind the group combining
// m with kb, which stays intact, the view gets n as a stride-0 dimension at
// the corresponding position, and no remark is emitted.
//
// CHECK-NOT: remark
// CHECK-LABEL: func.func @q80_q80_strict
// The a_d load chain: m (extent 64), kb (extent 1), then n (extent 32) with
// stride 0, read directly from the argument.
// CHECK: %[[ADV:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [64, 1, 32], strides: [1, 1, 0] {{.*}}stage = "compute"{{.*}} : memref<*xf16> to memref<64x1x32xf16, strided<[1, 1, 0]>>
// CHECK: %[[ADT:[a-z0-9_]+]] = bufferization.to_tensor %[[ADV]] restrict {{.*}}stage = "compute"{{.*}} : memref<64x1x32xf16, strided<[1, 1, 0]>> to tensor<64x1x32xf16>
// The grouped collapse_shape: [0, 1] is the original m/kb group, [2] the new
// group for n.
// CHECK: %[[ADC:[a-z0-9_]+]] = tensor.collapse_shape %[[ADT]] {{\[\[}}0, 1], [2]] {{.*}}stage = "compute"{{.*}} : tensor<64x1x32xf16> into tensor<64x32xf16>
// The extension of the factor runs in (m, n) space.
// CHECK: linalg.generic {{.*}}ins(%[[ADC]] : tensor<64x32xf16>) outs(%{{[a-z0-9_]+}} : tensor<64x32xf32>) {{.*}}stage = "compute"
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.extf
// CHECK: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark
// No load chain keeps a private copy anywhere in the output.
// NOCOPY-NOT: memref.alloc
// NOCOPY-NOT: memref.copy
