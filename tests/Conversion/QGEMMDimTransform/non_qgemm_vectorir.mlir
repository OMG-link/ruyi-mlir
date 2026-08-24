// RUN: timeout 120 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" --qgemm-tiling="tile-m=4 tile-n=4" --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs-from-loops=true" --lower-linalg-to-vir --lower-vir-to-vector="vector-width=4" --convert-linalg-to-affine-loops %S/../LinalgToVIR/linalg-to-vir-index-broadcast-regression.mlir 2>&1 | FileCheck --implicit-check-not=qgemm-recognize --implicit-check-not=qgemm.node --implicit-check-not=qgemm.root %s

// This input has no reduction root and is deliberately outside the QGEMM
// recognizer's domain. The complete vectorir lowering must still succeed,
// without a recognizer trace or diagnostic. The implicit negative checks
// cover the output before CHECK-LABEL as well as after it.
// CHECK-LABEL: func.func @index_and_zero_broadcast_regression
// CHECK: affine.for
// CHECK: memref.expand_shape
