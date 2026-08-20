// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q80_q80_layout_trans_reshape.ttshared.mlir > %t.trans.ir 2> %t.trans.trace
// RUN: FileCheck %s --check-prefix=TRANS-TRACE < %t.trans.trace
// RUN: FileCheck %s --check-prefix=TRANS-IR < %t.trans.ir
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --debug-only=qgemm-recognize %S/Inputs/q80_q80_layout_reshape_trans_reshape.ttshared.mlir > %t.reshape.ir 2> %t.reshape.trace
// RUN: FileCheck %s --check-prefix=RESHAPE-TRACE < %t.reshape.trace
// RUN: FileCheck %s --check-prefix=RESHAPE-IR < %t.reshape.ir
// RUN: grep '\[qgemm-recognize\] op' %t.trans.trace > %t.trans.walk
// RUN: grep '\[qgemm-recognize\] op' %t.reshape.trace > %t.reshape.walk
// RUN: diff -u %t.trans.walk %t.reshape.walk
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_q80_layout_trans_reshape.ttshared.mlir > %t.trans.direction
// RUN: FileCheck %s --check-prefix=TRANS-DIRECTION < %t.trans.direction
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_q80_layout_reshape_trans_reshape.ttshared.mlir > %t.reshape.direction
// RUN: FileCheck %s --check-prefix=RESHAPE-DIRECTION < %t.reshape.direction
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_q80_layout_trans_reshape.ttshared.mlir > %t.trans.raw
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_q80_layout_reshape_trans_reshape.ttshared.mlir > %t.reshape.raw
// RUN: sed -E -e '/^#loc/d' -e 's/ loc\(.*//' -e 's/@q80_q80_layout_[A-Za-z0-9_]+/@q80_q80_layout/g' -e 's/%([A-Za-z_][A-Za-z0-9_]*|[0-9]+)/%v/g' %t.trans.raw > %t.trans
// RUN: sed -E -e '/^#loc/d' -e 's/ loc\(.*//' -e 's/@q80_q80_layout_[A-Za-z0-9_]+/@q80_q80_layout/g' -e 's/%([A-Za-z_][A-Za-z0-9_]*|[0-9]+)/%v/g' %t.reshape.raw > %t.reshape
// RUN: diff -u %t.trans %t.reshape

// Compare the complete simplified IRs. Normalization removes only function
// names, SSA value names (including their numeric suffixes), inline source
// locations and standalone #loc definitions. Types, shapes, indexing maps,
// permutations and every operation name remain part of the comparison. The
// two walk traces are compared directly above, rather than normalizing their
// counts independently.
// TRANS-TRACE: [qgemm-recognize] candidates: 1
// TRANS-IR: linalg.transpose ins(%{{.*}} : tensor<16x32x1x32xi8>) outs(%{{.*}} : tensor<1x32x16x32xi8>) permutation = [2, 3, 0, 1]
// TRANS-IR: linalg.reduce ins(%{{.*}} : tensor<192x32x16xf32>) outs(%{{.*}} : tensor<192x32xf32>) dimensions = [2]
// RESHAPE-TRACE: [qgemm-recognize] candidates: 1
// RESHAPE-IR: linalg.transpose ins(%{{.*}} : tensor<16x32x1x32xi8>) outs(%{{.*}} : tensor<1x32x16x32xi8>) permutation = [2, 3, 0, 1]
// RESHAPE-IR: linalg.reduce ins(%{{.*}} : tensor<192x32x16xf32>) outs(%{{.*}} : tensor<192x32xf32>) dimensions = [2]

// Directionality is checked over every layout transform in the complete
// function body. For each transpose, no expand may consume its result. For
// each collapse, no transpose may consume its result and no expand may follow
// it directly. These checks deliberately use the printed SSA names only as
// FileCheck captures; the normalized IR comparison above remains independent
// of those names.
// TRANS-DIRECTION-LABEL: func.func @q80_q80_layout_trans_reshape
// TRANS-DIRECTION: %[[TRANSPOSE0:[A-Za-z0-9_]+]] = linalg.transpose
// TRANS-DIRECTION-NOT: tensor.expand_shape %[[TRANSPOSE0]]
// TRANS-DIRECTION: %[[TRANSPOSE1:[A-Za-z0-9_]+]] = linalg.transpose
// TRANS-DIRECTION-NOT: tensor.expand_shape %[[TRANSPOSE1]]
// TRANS-DIRECTION: %[[COLLAPSE0:[A-Za-z0-9_]+]] = tensor.collapse_shape
// TRANS-DIRECTION-NOT: linalg.transpose ins(%[[COLLAPSE0]]
// TRANS-DIRECTION-NOT: tensor.expand_shape %[[COLLAPSE0]]
// TRANS-DIRECTION-NOT: tensor.collapse_shape
// TRANS-DIRECTION: return

// The second spelling must satisfy the same three constraints over its
// complete output, not only over the weight-side chain used by TRANS-DIRECTION.
// RESHAPE-DIRECTION-LABEL: func.func @q80_q80_layout_reshape_trans_reshape
// RESHAPE-DIRECTION: %[[TRANSPOSE0:[A-Za-z0-9_]+]] = linalg.transpose
// RESHAPE-DIRECTION-NOT: tensor.expand_shape %[[TRANSPOSE0]]
// RESHAPE-DIRECTION: %[[TRANSPOSE1:[A-Za-z0-9_]+]] = linalg.transpose
// RESHAPE-DIRECTION-NOT: tensor.expand_shape %[[TRANSPOSE1]]
// RESHAPE-DIRECTION: %[[COLLAPSE0:[A-Za-z0-9_]+]] = tensor.collapse_shape
// RESHAPE-DIRECTION-NOT: linalg.transpose ins(%[[COLLAPSE0]]
// RESHAPE-DIRECTION-NOT: tensor.expand_shape %[[COLLAPSE0]]
// RESHAPE-DIRECTION-NOT: tensor.collapse_shape
// RESHAPE-DIRECTION: return
