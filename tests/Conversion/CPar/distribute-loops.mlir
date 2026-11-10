// RUN: buddy-opt %s --allow-unregistered-dialect --cpar-distribute-loops="num-threads=4" --split-input-file --verify-diagnostics | FileCheck %s
// RUN: buddy-opt %s --allow-unregistered-dialect --cpar-distribute-loops="num-threads=1" --split-input-file --verify-diagnostics | FileCheck %s --check-prefix=ONE

// A work-item forall is distributed round robin into a one iteration per
// thread forall: thread t runs iterations lb + t*step, lb + (t + N)*step, ...

// CHECK-LABEL: func.func @work_items
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index, %[[BUF:.*]]: memref<?xf32>)
// CHECK: scf.forall (%[[T:.*]]) in (4) {
// CHECK-NEXT: %[[C4:.*]] = arith.constant 4 : index
// CHECK-NEXT: %[[OFF:.*]] = arith.muli %[[T]], %[[STEP]] : index
// CHECK-NEXT: %[[NLB:.*]] = arith.addi %[[LB]], %[[OFF]] : index
// CHECK-NEXT: %[[NSTEP:.*]] = arith.muli %[[STEP]], %[[C4]] : index
// CHECK-NEXT: scf.for %[[IV:.*]] = %[[NLB]] to %[[UB]] step %[[NSTEP]] {
// CHECK-NEXT: memref.store %{{.*}}, %[[BUF]][%[[IV]]]
// CHECK-NEXT: } {tt.num_stages = 1 : i32}
// CHECK-NEXT: } {cpar.threads}
// CHECK-NEXT: return

// With one thread the forall becomes a plain sequential loop.
// ONE-LABEL: func.func @work_items
// ONE-NOT: scf.forall
// ONE: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// ONE-NOT: cpar
// ONE: return
func.func @work_items(%lb: index, %ub: index, %step: index, %buf: memref<?xf32>) {
  %c = arith.constant 1.0 : f32
  scf.forall (%iv) = (%lb) to (%ub) step (%step) {
    memref.store %c, %buf[%iv] : memref<?xf32>
  } {tt.num_stages = 1 : i32}
  return
}

// -----

// Static bounds are materialized as constants; ordinary loops are left alone.

// CHECK-LABEL: func.func @static_bounds
// CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK: scf.forall (%[[T:.*]]) in (4) {
// CHECK: arith.muli %[[T]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: } {cpar.threads}
func.func @static_bounds(%buf: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %f = arith.constant 2.0 : f32
  scf.for %j = %c0 to %c8 step %c1 {
    "test.serial"() : () -> ()
  }
  scf.forall (%i) = (0) to (1024) step (64) {
    memref.store %f, %buf[%i] : memref<?xf32>
  }
  return
}

// -----

// Only the outermost forall is distributed; the nested one runs sequentially
// in each thread.

// CHECK-LABEL: func.func @nested
// CHECK: scf.forall
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: scf.forall
// CHECK: {cpar.threads}
func.func @nested(%n: index, %buf: memref<?x?xf32>) {
  %f = arith.constant 3.0 : f32
  scf.forall (%i) in (%n) {
    // expected-remark @+1 {{nested parallel loop is executed sequentially by each thread}}
    scf.forall (%j) in (%n) {
      memref.store %f, %buf[%i, %j] : memref<?x?xf32>
    }
  }
  return
}

// -----

func.func @rank2(%n: index, %buf: memref<?x?xf32>) {
  %f = arith.constant 4.0 : f32
  // expected-error @+1 {{expected a 1-d scf.forall without shared_outs and mapping}}
  scf.forall (%i, %j) in (%n, %n) {
    memref.store %f, %buf[%i, %j] : memref<?x?xf32>
  }
  return
}
