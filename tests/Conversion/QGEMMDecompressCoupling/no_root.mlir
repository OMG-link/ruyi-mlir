// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre %s 2>&1 | FileCheck %s
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s
//
// Scope preservation, no-root half: a function without any `qgemm.root`
// passes through unchanged under both decompress values, and no remark is
// emitted. The expected output equals the input by Req 1.4, so this test is
// exempt from the recognize-only discrimination run (Req 12.9). Identity is
// checked as identity: the generic-form output of each mode is compared with
// the Topology_Checker against the same IR printed without the pass, stderr
// must be empty, and no qgemm attribute appears anywhere.
// RUN: buddy-opt --mlir-print-op-generic --mlir-print-local-scope %s -o %t.ref
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %s -o %t.pre 2> %t.pre.err
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %s -o %t.otf 2> %t.otf.err
// RUN: %python %S/Inputs/topo_equal.py %t.ref %t.pre
// RUN: %python %S/Inputs/topo_equal.py %t.ref %t.otf
// RUN: FileCheck %s --check-prefix=ERR --allow-empty --input-file=%t.pre.err
// RUN: FileCheck %s --check-prefix=ERR --allow-empty --input-file=%t.otf.err
// RUN: FileCheck %s --check-prefix=NOQ --input-file=%t.pre
// RUN: FileCheck %s --check-prefix=NOQ --input-file=%t.otf
// ERR-NOT: {{.}}
// NOQ-NOT: qgemm
// NOQ: sym_name = "no_root"
// NOQ-NOT: qgemm

// CHECK-NOT: remark
// CHECK-LABEL: func.func @no_root
func.func @no_root(%a: tensor<4x8xi8>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %e = tensor.empty() : tensor<4x8xf32>
  // CHECK: %[[CVT:.+]] = linalg.generic
  // CHECK: arith.sitofp
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<4x8xi8>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %f = arith.sitofp %in : i8 to f32
    linalg.yield %f : f32
  } -> tensor<4x8xf32>
  // CHECK: %[[MUL:.+]] = linalg.generic
  // CHECK-SAME: ins(%[[CVT]], %{{[a-z0-9_]+}} : tensor<4x8xf32>, tensor<4x8xf32>)
  // CHECK: arith.mulf
  %m = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%c, %b : tensor<4x8xf32>, tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %p = arith.mulf %x, %y : f32
    linalg.yield %p : f32
  } -> tensor<4x8xf32>
  // CHECK: return %[[MUL]]
  // CHECK-NOT: linalg.broadcast
  // CHECK-NOT: remark
  return %m : tensor<4x8xf32>
}
