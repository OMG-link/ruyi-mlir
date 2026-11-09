// RUN: buddy-opt %s --cpu-tcm-assign --split-input-file --verify-diagnostics | FileCheck %s
// RUN: buddy-opt %s --cpu-tcm-assign --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts --split-input-file --verify-diagnostics | FileCheck %s --check-prefix=LLVM

// Buffers of triton.language.extra.cpu.alloc (memref.alloc {cpu.tcm}) get
// static offsets in the TCM of the executing thread, in program order, each
// aligned.  %a is used inside the region (shared by its threads), %b only by
// the thread that allocated it, %c is allocated by every thread of the region.

// CHECK-DAG:   llvm.func @triton_cpu_tcm_base(i64) -> !llvm.ptr
// CHECK-DAG:   llvm.func @triton_cpu_tcm_base_shared(i64) -> !llvm.ptr
// CHECK-LABEL: func.func @kernel
// CHECK:         %[[TOTAL:.*]] = llvm.mlir.constant(288 : i64) : i64
// CHECK-NEXT:    %[[A_BASE:.*]] = llvm.call @triton_cpu_tcm_base_shared(%[[TOTAL]]) : (i64) -> !llvm.ptr
// CHECK-NOT:     llvm.getelementptr
// CHECK:         %[[A:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<100xi8>
// CHECK:         %[[B_BASE:.*]] = llvm.call @triton_cpu_tcm_base(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:         llvm.getelementptr %[[B_BASE]][128] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:         %[[B:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<64xi8>
// CHECK:         memref.view %[[B]]
// CHECK:         cpar.parallel num_threads(2)
// CHECK:           memref.view %[[A]]
// CHECK:           %[[C_BASE:.*]] = llvm.call @triton_cpu_tcm_base(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:           llvm.getelementptr %[[C_BASE]][256] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NOT:     memref.alloc

// LLVM-LABEL:  llvm.func @kernel
// LLVM:          llvm.call @triton_cpu_tcm_base_shared(
// LLVM:          llvm.call @triton_cpu_tcm_base(
// LLVM-NOT:      unrealized_conversion_cast
// LLVM-NOT:      llvm.call @malloc
func.func @kernel(%x: f32, %i: index) {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() {alignment = 128 : i64, cpu.tcm} : memref<100xi8>
  %b = memref.alloc() {alignment = 64 : i64, cpu.tcm} : memref<64xi8>
  %bv = memref.view %b[%c0][] : memref<64xi8> to memref<16xf32>
  memref.store %x, %bv[%i] : memref<16xf32>
  cpar.parallel num_threads(2) {
  ^bb0(%tid: i64, %nt: i64):
    %av = memref.view %a[%c0][] : memref<100xi8> to memref<25xf32>
    %v = memref.load %av[%i] : memref<25xf32>
    %c = memref.alloc() {alignment = 128 : i64, cpu.tcm} : memref<32xi8>
    %cv = memref.view %c[%c0][] : memref<32xi8> to memref<8xf32>
    memref.store %v, %cv[%i] : memref<8xf32>
    cpar.yield
  }
  return
}

// -----

// Ordinary allocations are left alone.

// CHECK-LABEL: func.func @plain
// CHECK:         memref.alloc() : memref<16xf32>
// CHECK-NOT:     llvm.call
// LLVM-LABEL:  llvm.func @plain
func.func @plain() -> memref<16xf32> {
  %m = memref.alloc() : memref<16xf32>
  return %m : memref<16xf32>
}

// -----

// expected-error @+1 {{cpu.alloc: the buffers of this kernel need 393217 bytes of TCM, more than the 393216 bytes available to a thread}}
module {
  func.func @too_big() {
    // expected-note @+1 {{393216 bytes at offset 0}}
    %a = memref.alloc() {alignment = 128 : i64, cpu.tcm} : memref<393216xi8>
    // expected-note @+1 {{1 bytes at offset 393216}}
    %b = memref.alloc() {alignment = 128 : i64, cpu.tcm} : memref<1xi8>
    return
  }
}
