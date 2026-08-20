// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q40_q80_canonical.ttshared.mlir | FileCheck %s

// The Q4_0 packed-byte assembly uses a tensor.insert_slice chain. The
// simplifier preserves both slice positions and their 32x16x16x2 target
// shape, makes the transpose consume the second slice result directly, and
// does not insert a tensor.expand_shape introducing a unit dimension between
// that slice and the transpose.
// CHECK-LABEL: func.func @q40_q80_canonical
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0, 0, 0] [32, 16, 16, 1] [1, 1, 1, 1] : tensor<32x16x16xi8> into tensor<32x16x16x2xi8>
// CHECK-NEXT: %[[SLICE:[A-Za-z0-9_]+]] = tensor.insert_slice %{{.*}} into %{{.*}}[0, 0, 0, 1] [32, 16, 16, 1] [1, 1, 1, 1] : tensor<32x16x16xi8> into tensor<32x16x16x2xi8>
// CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<32x16x2x16xi8>
// CHECK-NOT: tensor.expand_shape
// CHECK-NEXT: %{{.*}} = linalg.transpose ins(%[[SLICE]] : tensor<32x16x16x2xi8>) outs(%{{.*}} : tensor<32x16x2x16xi8>) permutation = [0, 1, 3, 2]
