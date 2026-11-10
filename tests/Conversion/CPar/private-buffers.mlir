// RUN: buddy-opt %s --cpar-distribute-loops="num-threads=4" --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs-from-loops=true" --cpar-forall-to-parallel | FileCheck %s
// RUN: buddy-opt %s --cpar-distribute-loops="num-threads=1" --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs-from-loops=true" | FileCheck %s --check-prefix=SEQ

// Reduced from a Triton kernel: the destination tensor %e is defined outside
// of the parallel loop (CSE hoists it) and written in every iteration.
//
// Sequentially, bufferization may reuse one buffer for all iterations (second
// RUN line: the allocation stays outside of the loop).  Executed by threads,
// that buffer would be shared: a data race.  Distributing at tensor level
// into a thread forall makes the write out of place inside the parallel
// region, so every thread allocates its own buffer.

// SEQ-LABEL: func.func @kernel
// SEQ: memref.alloc
// SEQ: scf.for

// CHECK-LABEL: func.func @kernel
// CHECK-SAME: (%[[OUT:.*]]: memref<?xf32>, %[[N:.*]]: index)
// CHECK-NOT: memref.alloc
// CHECK: cpar.parallel num_threads(4) {
// CHECK: ^bb0(%[[TID:.*]]: i64, %{{.*}}: i64):
// CHECK: scf.for
// CHECK: %[[PRIV:.*]] = memref.alloc() {{.*}} : memref<64xf32>
// CHECK: linalg.generic {{.*}} outs(%[[PRIV]] : memref<64xf32>)
// CHECK: memref.copy %[[PRIV]]
// CHECK: cpar.yield
// CHECK-NOT: scf.forall
#map = affine_map<(d0) -> (d0)>
func.func @kernel(%out: memref<?xf32>, %n: index) {
  %e = tensor.empty() : tensor<64xf32>
  scf.forall (%i) = (0) to (%n) step (64) {
    %f = arith.index_cast %i : index to i64
    %v = arith.sitofp %f : i64 to f32
    %t = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]}
        outs(%e : tensor<64xf32>) {
      ^bb0(%o: f32):
        linalg.yield %v : f32
    } -> tensor<64xf32>
    %dst = memref.subview %out[%i] [64] [1]
        : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %t in writable %dst
        : (tensor<64xf32>, memref<64xf32, strided<[1], offset: ?>>) -> ()
  }
  return
}
