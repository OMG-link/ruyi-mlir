// RUN: buddy-opt %s --cpar-outline-to-runtime --split-input-file --verify-diagnostics | FileCheck %s

// Captured values go through a context struct allocated in the function's
// entry block; constants are cloned into the outlined function.

// CHECK: llvm.func @__cpar_parallel(!llvm.ptr, !llvm.ptr, i64)
// CHECK-LABEL: llvm.func internal @kernel__cpar0
// CHECK-SAME: (%[[CTX:.*]]: !llvm.ptr, %[[TID:.*]]: i64, %[[NT:.*]]: i64)
// CHECK-DAG: %[[C:.*]] = llvm.mlir.constant(7 : i64) : i64
// Fields follow the order of first use in the region: %s, then %a.
// CHECK-DAG: %[[P0:.*]] = llvm.getelementptr %[[CTX]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, ptr)>
// CHECK-DAG: %[[SCALE:.*]] = llvm.load %[[P0]] : !llvm.ptr -> i64
// CHECK-DAG: %[[P1:.*]] = llvm.getelementptr %[[CTX]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, ptr)>
// CHECK-DAG: %[[BASE:.*]] = llvm.load %[[P1]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.br ^bb1(%[[TID]], %[[NT]] : i64, i64)
// CHECK: ^bb1(%[[T:.*]]: i64, %{{.*}}: i64):
// CHECK: %[[V:.*]] = llvm.mul %[[T]], %[[SCALE]]
// CHECK: llvm.add %[[V]], %[[C]]
// CHECK: llvm.getelementptr %[[BASE]][%[[T]]]
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @kernel
// CHECK-SAME: (%[[A:.*]]: !llvm.ptr, %[[S:.*]]: i64)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[CTXP:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(i64, ptr)>
// CHECK: %[[F0:.*]] = llvm.getelementptr %[[CTXP]][0, 0]
// CHECK-NEXT: llvm.store %[[S]], %[[F0]] : i64, !llvm.ptr
// CHECK-NEXT: %[[F1:.*]] = llvm.getelementptr %[[CTXP]][0, 1]
// CHECK-NEXT: llvm.store %[[A]], %[[F1]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[FN:.*]] = llvm.mlir.addressof @kernel__cpar0 : !llvm.ptr
// CHECK-NEXT: %[[N:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: llvm.call @__cpar_parallel(%[[FN]], %[[CTXP]], %[[N]])
// CHECK-NOT: cpar.
// CHECK: llvm.return
llvm.func @kernel(%a: !llvm.ptr, %s: i64) {
  %c7 = llvm.mlir.constant(7 : i64) : i64
  %f = llvm.mlir.constant(1.0 : f32) : f32
  cpar.parallel num_threads(4) {
  ^bb0(%tid: i64, %nt: i64):
    %v = llvm.mul %tid, %s : i64
    %w = llvm.add %v, %c7 : i64
    %p = llvm.getelementptr %a[%tid] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %wf = llvm.sitofp %w : i64 to f32
    %x = llvm.fadd %wf, %f : f32
    llvm.store %x, %p : f32, !llvm.ptr
    cpar.yield
  }
  llvm.return
}

// -----

// A region without captures passes a null context.  Multi-block regions keep
// their control flow; every cpar.yield becomes llvm.return.

// CHECK-LABEL: llvm.func internal @f__cpar0
// CHECK: llvm.cond_br
// CHECK: llvm.return
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @f
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @__cpar_parallel(%{{.*}}, %[[NULL]], %{{.*}})
llvm.func @f() {
  cpar.parallel num_threads(2) {
  ^bb0(%tid: i64, %nt: i64):
    %zero = llvm.mlir.constant(0 : i64) : i64
    %c = llvm.icmp "eq" %tid, %zero : i64
    llvm.cond_br %c, ^bb1, ^bb2
  ^bb1:
    cpar.yield
  ^bb2:
    cpar.yield
  }
  llvm.return
}

// -----

// Nested regions: the inner one is outlined first, its context is allocated
// in the outer region (one context per outer thread) and captured by it.

// CHECK-LABEL: llvm.func internal @g__cpar0
// CHECK: llvm.load
// CHECK: llvm.return
// CHECK-LABEL: llvm.func internal @g__cpar1
// CHECK: llvm.br ^bb1
// CHECK: ^bb1(%[[OT:.*]]: i64, %{{.*}}: i64):
// CHECK-NEXT: llvm.mlir.constant(1 : i64)
// CHECK-NEXT: llvm.alloca
// CHECK: llvm.store %[[OT]]
// CHECK: llvm.mlir.addressof @g__cpar0
// CHECK: llvm.call @__cpar_parallel
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @g
// CHECK: llvm.mlir.addressof @g__cpar1
// CHECK: llvm.call @__cpar_parallel
llvm.func @g(%a: !llvm.ptr) {
  cpar.parallel num_threads(2) {
  ^bb0(%otid: i64, %ont: i64):
    cpar.parallel num_threads(3) {
    ^bb0(%itid: i64, %int: i64):
      %s = llvm.add %otid, %itid : i64
      %p = llvm.getelementptr %a[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      llvm.store %s, %p : i64, !llvm.ptr
      cpar.yield
    }
    cpar.yield
  }
  llvm.return
}

// -----

func.func @not_llvm(%m: memref<4xf32>) {
  // expected-error @+1 {{must be nested in an llvm.func}}
  cpar.parallel num_threads(2) {
  ^bb0(%tid: i64, %nt: i64):
    cpar.yield
  }
  return
}
