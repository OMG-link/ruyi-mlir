// RUN: buddy-opt %s --cpar-forall-to-parallel --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @convert
// CHECK-NOT: scf.forall
// CHECK: cpar.parallel num_threads(3) {
// CHECK-NEXT: ^bb0(%[[TID:.*]]: i64, %{{.*}}: i64):
// CHECK-NEXT: %[[T:.*]] = arith.index_cast %[[TID]] : i64 to index
// CHECK-NEXT: memref.store %{{.*}}, %{{.*}}[%[[T]]]
// CHECK-NEXT: cpar.yield
func.func @convert(%buf: memref<3xf32>) {
  %f = arith.constant 1.0 : f32
  scf.forall (%t) in (3) {
    memref.store %f, %buf[%t] : memref<3xf32>
  } {cpar.threads}
  return
}

// -----

// Foralls without the marker are not touched.

// CHECK-LABEL: func.func @unmarked
// CHECK: scf.forall
// CHECK-NOT: cpar.parallel
func.func @unmarked(%buf: memref<3xf32>) {
  %f = arith.constant 1.0 : f32
  scf.forall (%t) in (3) {
    memref.store %f, %buf[%t] : memref<3xf32>
  }
  return
}

// -----

func.func @tensor_results(%t: tensor<3xf32>) -> tensor<3xf32> {
  // expected-error @+1 {{the forall still has tensor results; run it after bufferization}}
  %r = scf.forall (%i) in (3) shared_outs(%o = %t) -> tensor<3xf32> {
    scf.forall.in_parallel {
    }
  } {cpar.threads}
  return %r : tensor<3xf32>
}
