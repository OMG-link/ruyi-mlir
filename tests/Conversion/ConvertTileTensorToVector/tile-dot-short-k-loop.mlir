// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -convert-tile-tensor-to-vector=allow-ime-tile-contracts -canonicalize | FileCheck %s
//
// A K-short tile product whose B is loaded before a loop and carried as an
// iter_arg through a cpu.sched_barrier converts with the carry. Without an
// fp16 tile (the K1) the barrier operand cannot convert, which is an error,
// so this case runs on the A100 only.

// CHECK-LABEL: func.func @loop_carried_b
// CHECK: scf.for {{.+}} -> (vector<8x1xf16>)
// CHECK: cpu.sched_barrier %{{.+}} : vector<8x1xf16>
// CHECK: vector.contract {{.+}} : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
// CHECK-NOT: linalg.matmul
#id = affine_map<(i, j) -> (i, j)>
func.func @loop_carried_b(%a: memref<8x1xf16, strided<[8, 1]>>,
                          %b: memref<8x1xf16, strided<[8, 1]>>,
                          %c: memref<8x8xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x8xf32>
  %z = linalg.fill ins(%zero : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
  %tb = bufferization.to_tensor %b restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%wb = %tb) -> (tensor<8x1xf16>) {
    %held = cpu.sched_barrier %wb : tensor<8x1xf16>
    %ta = bufferization.to_tensor %a restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
    %eh = tensor.empty() : tensor<1x8xf16>
    %bt = linalg.transpose ins(%held : tensor<8x1xf16>) outs(%eh : tensor<1x8xf16>) permutation = [1, 0]
    %ea0 = tensor.empty() : tensor<8x1xf32>
    %ea = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%ta : tensor<8x1xf16>) outs(%ea0 : tensor<8x1xf32>) {
    ^bb0(%x: f16, %y: f32):
      %v = arith.extf %x : f16 to f32
      linalg.yield %v : f32
    } -> tensor<8x1xf32>
    %eb0 = tensor.empty() : tensor<1x8xf32>
    %eb = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%bt : tensor<1x8xf16>) outs(%eb0 : tensor<1x8xf32>) {
    ^bb0(%x: f16, %y: f32):
      %v = arith.extf %x : f16 to f32
      linalg.yield %v : f32
    } -> tensor<1x8xf32>
    %m = linalg.matmul ins(%ea, %eb : tensor<8x1xf32>, tensor<1x8xf32>) outs(%z : tensor<8x8xf32>) -> tensor<8x8xf32>
    bufferization.materialize_in_destination %m in writable %c : (tensor<8x8xf32>, memref<8x8xf32>) -> ()
    scf.yield %held : tensor<8x1xf16>
  }
  return
}
