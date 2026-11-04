// RUN: buddy-opt --allow-unregistered-dialect %s | buddy-opt --allow-unregistered-dialect | FileCheck %s

// Round trip of a single-block and of a multi-block region.

// CHECK-LABEL: func.func @single_block
// CHECK: cpar.parallel num_threads(4) {
// CHECK-NEXT: ^bb0(%[[TID:.*]]: i64, %[[NT:.*]]: i64):
// CHECK-NEXT: "test.use"(%[[TID]], %[[NT]])
// CHECK-NEXT: cpar.yield
func.func @single_block() {
  cpar.parallel num_threads(4) {
  ^bb0(%tid: i64, %nt: i64):
    "test.use"(%tid, %nt) : (i64, i64) -> ()
    cpar.yield
  }
  return
}

// CHECK-LABEL: func.func @multi_block
// CHECK: cpar.parallel num_threads(2) {
// CHECK: cf.cond_br
// CHECK: cpar.yield
// CHECK: cpar.yield
func.func @multi_block(%c: i1) {
  cpar.parallel num_threads(2) {
  ^bb0(%tid: i64, %nt: i64):
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    cpar.yield
  ^bb2:
    cpar.yield
  }
  return
}
