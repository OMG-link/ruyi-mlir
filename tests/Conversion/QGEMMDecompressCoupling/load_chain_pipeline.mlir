// RUN: buddy-opt %S/Inputs/q80_q80_strict_noalias.ttshared.mlir --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --qgemm-tiling="tile-m=32 tile-n=32" --qgemm-tiling="tile-m=4 tile-n=4" --empty-tensor-to-alloc-tensor --one-shot-bufferize=allow-return-allocs-from-loops=true --lower-linalg-to-vir --lower-vir-to-vector=vector-width=4 --convert-linalg-to-affine-loops 2>&1 | FileCheck %s
//
// Req 12.23, 14.10: the lifted load chain survives the rest of the `vectorir`
// stage. The pass list is the one `_build_vectorir_passes` in
// triton-riscv/backend/compiler.py builds with qgemm_opt_enable set: both
// --qgemm-tiling rounds, bufferization, the VIR lowering and the vector
// lowering. The run fails on a verifier error, a crash or a non-zero exit,
// since nothing would reach FileCheck.
//
// What the stride-0 view becomes: bufferization turns the tiled slice chain
// into nested memref.subview of the view itself, with no copy of the lifted
// operand, so the n dimension still reads one row of the argument for every
// index along it.
// CHECK-LABEL: func.func @q80_q80_strict
// CHECK: %[[V:[a-z0-9_]+]] = memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [192, 32, 512], strides: [512, 0, 1] {{.*}} : memref<*xi8> to memref<192x32x512xi8, strided<[512, 0, 1]>>
// The reduction-dimension reshape became a memref view of the same buffer.
// CHECK: %[[E:[a-z0-9_]+]] = memref.expand_shape %[[V]] {{.*}} : memref<192x32x512xi8, strided<[512, 0, 1]>> into memref<192x32x16x32xi8, strided<[512, 0, 32, 1]>>
// CHECK: scf.for
// CHECK: memref.subview %[[E]]
// No copy of a lifted operand is reintroduced anywhere in the kernel. Both
// lifted value chains are i8, and no other i8 buffer exists after
// bufferization, so no memref.copy of an i8 memref may appear. The f32 copies
// that do remain belong to the tiled accumulator and the output store, which
// this feature does not touch.
// CHECK-NOT: memref.copy {{.*}}xi8
