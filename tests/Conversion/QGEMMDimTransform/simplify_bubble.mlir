// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/collapse_transpose_expand.mlir | FileCheck %s

// The final reduction's tensor-semantics fill is a root-neutral boundary.
// Consequently, the output-layout chain feeding that fill is outside the
// coarse QGEMM scope and remains in its original shape order.
// CHECK-LABEL: func.func @collapse_transpose_expand
// CHECK: %[[COLLAPSE:[A-Za-z0-9_]+]] = tensor.collapse_shape %{{.*}} {{\[\[0, 1\], \[2\]\]}} : tensor<2x3x4xf32> into tensor<6x4xf32>
// CHECK: %[[TRANSPOSE:[A-Za-z0-9_]+]] = linalg.transpose ins(%[[COLLAPSE]] : tensor<6x4xf32>) outs(%{{.*}} : tensor<4x6xf32>) permutation = [1, 0]
// CHECK: tensor.expand_shape %[[TRANSPOSE]] {{\[\[0, 1\], \[2\]\]}} output_shape [2, 2, 6]
// CHECK: tensor.insert_slice
// CHECK: tensor.extract_slice
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[0, 1\], \[2\]\]}} : tensor<2x2x6xf32> into tensor<4x6xf32>
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x6xf32>) -> tensor<4x6xf32>

// The reduction data operand stays inside the coarse QGEMM scope. The pass
// bubbles its collapse -> transpose -> expand chain into expand -> transpose
// -> collapse before the final reduction.
// CHECK-LABEL: func.func @scoped_collapse_transpose_expand
// CHECK: %[[SCOPED_TRANSPOSE:[A-Za-z0-9_]+]] = linalg.transpose ins(%{{.*}} : tensor<2x3x2x2xf32>) outs(%{{.*}} : tensor<2x2x2x3xf32>) permutation = [2, 3, 0, 1]
// CHECK-NOT: tensor.expand_shape %[[SCOPED_TRANSPOSE]]
// CHECK: %[[SCOPED_COLLAPSE:[A-Za-z0-9_]+]] = tensor.collapse_shape %[[SCOPED_TRANSPOSE]] {{\[\[0\], \[1\], \[2, 3\]\]}} : tensor<2x2x2x3xf32> into tensor<2x2x6xf32>
// CHECK-NOT: linalg.transpose ins(%[[SCOPED_COLLAPSE]]
// CHECK-NOT: tensor.expand_shape %[[SCOPED_COLLAPSE]]
// CHECK: linalg.reduce ins(%[[SCOPED_COLLAPSE]] : tensor<2x2x6xf32>)
