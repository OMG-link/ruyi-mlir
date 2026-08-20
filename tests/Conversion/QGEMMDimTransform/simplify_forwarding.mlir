// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_one_candidate.mlir | FileCheck %s
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir | FileCheck %s --check-prefix=STRICT
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q40_q80_canonical.ttshared.mlir > %t.q40.direction
// RUN: FileCheck %s --check-prefix=Q40-DIRECTION < %t.q40.direction
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir > %t.q80.direction
// RUN: FileCheck %s --check-prefix=Q80-DIRECTION < %t.q80.direction

// Forwarding moves the reduction-block reshape through the element-wise
// multiply. The resulting multiply generic is immediately followed by its
// reduction, rather than by a standalone tensor.expand_shape.
// CHECK-LABEL: func.func @q80_one_candidate
// CHECK: %[[MUL:[A-Za-z0-9_]+]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<8x8x1x32xf32>, tensor<8x8x1x32xf32>) outs(%{{.*}} : tensor<8x8x1x32xf32>)
// CHECK: arith.mulf
// CHECK: } -> tensor<8x8x1x32xf32>
// CHECK-NOT: tensor.expand_shape
// CHECK: %{{.*}} = linalg.reduce ins(%[[MUL]] : tensor<8x8x1x32xf32>)

// The real Q8_0 capture must also leave the reduction-block reshape folded
// into the multiply generic. The concrete tensor type prevents this check
// from accepting a different reshape or deleting the surrounding IR.
// STRICT-LABEL: func.func @q80_q80_strict
// STRICT: %[[STRICT_MUL:[A-Za-z0-9_]+]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<192x32x16x32xi16>, tensor<192x32x16x32xi16>) outs(%{{.*}} : tensor<192x32x16x32xi16>)
// STRICT: arith.muli
// STRICT: } -> tensor<192x32x16x32xi16>
// STRICT-NOT: tensor.expand_shape
// STRICT: linalg.generic {{.*}}tensor<192x32x16x32xi32>
// STRICT-NOT: tensor.expand_shape
// STRICT: linalg.reduce ins(%{{[A-Za-z0-9_]+}} : tensor<192x32x16x32xi32>)

// A projected indexing map is the normalized broadcast spelling. Its result
// must be consumed by the next Linalg op, not by a direct tensor.expand_shape.
// Q40-DIRECTION-LABEL: func.func @q40_q80_canonical
// Q40-DIRECTION: %[[Q40_BROADCAST:[A-Za-z0-9_]+]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<192x16x32xi16>) outs(%{{.*}} : tensor<192x32x16x32xi16>)
// Q40-DIRECTION-NOT: tensor.expand_shape %[[Q40_BROADCAST]]
// Q40-DIRECTION: arith.muli

// The same directionality invariant holds for the strict Q8_0 capture.
// Q80-DIRECTION-LABEL: func.func @q80_q80_strict
// Q80-DIRECTION: %[[Q80_BROADCAST:[A-Za-z0-9_]+]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<192x16x32xi16>) outs(%{{.*}} : tensor<192x32x16x32xi16>)
// Q80-DIRECTION-NOT: tensor.expand_shape %[[Q80_BROADCAST]]
// Q80-DIRECTION: arith.muli
