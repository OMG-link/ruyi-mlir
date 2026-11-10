// RUN: buddy-opt %s --allow-unregistered-dialect --cpar-distribute-loops="num-threads=4" --split-input-file --verify-diagnostics | FileCheck %s
// RUN: buddy-opt %s --allow-unregistered-dialect --cpar-distribute-loops="num-threads=1" --split-input-file --verify-diagnostics | FileCheck %s --check-prefix=ONE

// A marked i32 loop (the type Triton uses) is distributed round robin into a
// one iteration per thread forall: thread t runs iterations lb + t*step,
// lb + (t + N)*step, ...

// CHECK-LABEL: func.func @marked_i32
// CHECK-SAME: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32, %[[BUF:.*]]: memref<?xf32>)
// CHECK: scf.forall (%[[T:.*]]) in (4) {
// CHECK-NEXT: %[[TID:.*]] = arith.index_cast %[[T]] : index to i32
// CHECK-NEXT: %[[C4:.*]] = arith.constant 4 : index
// CHECK-NEXT: %[[NT:.*]] = arith.index_cast %[[C4]] : index to i32
// CHECK-NEXT: %[[OFF:.*]] = arith.muli %[[TID]], %[[STEP]] : i32
// CHECK-NEXT: %[[NLB:.*]] = arith.addi %[[LB]], %[[OFF]] : i32
// CHECK-NEXT: %[[NSTEP:.*]] = arith.muli %[[STEP]], %[[NT]] : i32
// CHECK-NEXT: scf.for %[[IV:.*]] = %[[NLB]] to %[[UB]] step %[[NSTEP]] : i32 {
// CHECK-NEXT: %[[IDX:.*]] = arith.index_cast %[[IV]]
// CHECK-NEXT: memref.store %{{.*}}, %[[BUF]][%[[IDX]]]
// CHECK-NEXT: } {tt.num_stages = 1 : i32}
// CHECK-NEXT: } {cpar.threads}
// CHECK-NEXT: return

// ONE-LABEL: func.func @marked_i32
// ONE-NOT: scf.forall
// ONE: scf.for
// ONE-NOT: cpar.parallel
// ONE: return
func.func @marked_i32(%lb: i32, %ub: i32, %step: i32, %buf: memref<?xf32>) {
  %c = arith.constant 1.0 : f32
  scf.for %iv = %lb to %ub step %step : i32 {
    %i = arith.index_cast %iv : i32 to index
    memref.store %c, %buf[%i] : memref<?xf32>
  } {cpar.parallel, tt.num_stages = 1 : i32}
  return
}

// -----

// Index loops need no cast; unmarked loops are left alone.

// CHECK-LABEL: func.func @marked_index
// CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK: scf.forall (%[[T:.*]]) in (4) {
// CHECK-NOT: arith.index_cast
// CHECK: arith.muli %[[T]]
// CHECK: scf.for
// CHECK: } {cpar.threads}
func.func @marked_index(%n: index, %buf: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %f = arith.constant 2.0 : f32
  scf.for %j = %c0 to %c8 step %c1 {
    "test.serial"() : () -> ()
  }
  scf.for %i = %c0 to %n step %c1 {
    memref.store %f, %buf[%i] : memref<?xf32>
  } {cpar.parallel}
  return
}

// -----

// Only the outermost marked loop is distributed.

// CHECK-LABEL: func.func @nested
// CHECK: scf.forall
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: cpar.parallel
// CHECK: {cpar.threads}
func.func @nested(%n: index, %buf: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 3.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    // expected-remark @+1 {{nested parallel loop is executed sequentially by each thread}}
    scf.for %j = %c0 to %n step %c1 {
      memref.store %f, %buf[%i, %j] : memref<?x?xf32>
    } {cpar.parallel}
  } {cpar.parallel}
  return
}

// -----

func.func @carried(%n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  // expected-error @+1 {{a parallel loop must not carry values from one iteration to the next}}
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> f32 {
    %next = arith.addf %acc, %acc : f32
    scf.yield %next : f32
  } {cpar.parallel}
  return %r : f32
}
