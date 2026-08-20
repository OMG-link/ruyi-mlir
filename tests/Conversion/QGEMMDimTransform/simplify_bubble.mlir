// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/collapse_transpose_expand.mlir | FileCheck %s

// The third exchange starts as collapse -> transpose -> expand. With the
// expand-consumer direction enabled, the stable result keeps the unit-extent
// expansion before the transpose and leaves the following collapse in place.
// The complete ordering still asserts that no expand directly consumes the
// transpose, that no transpose directly consumes a collapse, and that no
// collapse is followed by an expand.
// CHECK-LABEL: func.func @collapse_transpose_expand
// CHECK: %[[TRANSPOSE:[A-Za-z0-9_]+]] = linalg.transpose ins(%{{.*}} : tensor<2x3x2x2xf32>) outs(%{{.*}} : tensor<2x2x2x3xf32>) permutation = [2, 3, 0, 1]
// CHECK-NOT: tensor.expand_shape %[[TRANSPOSE]]
// CHECK: %[[COLLAPSE:[A-Za-z0-9_]+]] = tensor.collapse_shape %[[TRANSPOSE]] {{\[\[0\], \[1\], \[2, 3\]\]}} : tensor<2x2x2x3xf32> into tensor<2x2x6xf32>
// CHECK-NOT: linalg.transpose ins(%[[COLLAPSE]]
// CHECK-NOT: tensor.expand_shape %[[COLLAPSE]]
