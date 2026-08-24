// RUN: timeout 120 buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-tiling="tile-m=32 tile-n=32" --qgemm-tiling="tile-m=4 tile-n=4" --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs-from-loops=true" --lower-linalg-to-vir --lower-vir-to-vector="vector-width=4" --convert-linalg-to-affine-loops %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir | FileCheck %s

// The LinalgToVIR input transform creates a memref.expand_shape with an
// entirely static output shape inside vir.set_vl. getOutputShape() returns
// only dynamic operands, so the old VIRToVector code saw an empty list and
// built a rank-mismatched clone. getMixedOutputShape() preserves the static
// bounds (and any dynamic operands) needed by the cloned operation.
// CHECK: memref.expand_shape {{.*}} output_shape [1,
