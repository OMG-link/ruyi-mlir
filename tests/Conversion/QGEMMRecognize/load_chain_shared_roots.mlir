// RUN: buddy-opt --mlir-disable-threading --debug-only=qgemm-recognize --qgemm-recognize %s > %t.ir 2> %t.stderr
// RUN: FileCheck %s --check-prefix=DIAG --implicit-check-not=shared-non-boundary-operation --implicit-check-not=error: --implicit-check-not=warning: < %t.stderr
// RUN: FileCheck %s --check-prefix=TRACE < %t.stderr
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.ir
//
// Requirement 14.4, 14.5 and 12.19: one load chain feeding two roots.
//
// Both products read the same bufferization.to_tensor, so every op of that
// chain (the memref.reinterpret_cast, the memref.alloc, the memref.copy and
// the to_tensor itself) is reachable from two roots. Each is demoted to a
// root-neutral boundary and reported exactly once with Remark R4 naming both
// roots; neither candidate is rejected, so the ownership check never reports
// shared-non-boundary-operation (excluded through the RUN line). The
// unshared chain of each rhs stays owned by its own root.

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast_m = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast_n = affine_map<(d0, d1, d2) -> (0, d1, d2)>

// Exactly four R4 remarks, one per shared chain op, each naming both roots.
// DIAG-COUNT-4: remark: qgemm-recognize: load-chain operation is shared by roots 0, 1; left root-neutral instead of rejecting the candidates
// DIAG-NOT: remark: qgemm-recognize: load-chain operation is shared

// Both candidates are still accepted.
// TRACE: [qgemm-recognize] op 0 accepted: root=0
// TRACE: [qgemm-recognize] op 1 accepted: root=1

// IR-LABEL: func.func @shared_load_chain
// The four shared chain ops carry no qgemm.node.
// IR: memref.reinterpret_cast %{{[a-z0-9]+}} to offset: [0], sizes: [4, 1, 4], strides: [4, 4, 1] : memref<*xf32> to memref<4x1x4xf32, strided<[4, 4, 1]>>{{ *$}}
// IR: memref.alloc() : memref<4x1x4xf32>{{ *$}}
// IR: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<4x1x4xf32, strided<[4, 4, 1]>> to memref<4x1x4xf32>{{ *$}}
// IR: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable : memref<4x1x4xf32> to tensor<4x1x4xf32>{{ *$}}
// Each rhs has its own chain, owned by the root that reads it.
// IR: memref.alloc() {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : memref<1x4x4xf32>
// IR: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : memref<1x4x4xf32> to tensor<1x4x4xf32>
// IR: linalg.generic{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}
// IR: linalg.reduce{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// IR: memref.alloc() {qgemm.node = {root = 1 : i64, stage = "data_prep"}} : memref<1x4x4xf32>
// IR: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {qgemm.node = {root = 1 : i64, stage = "data_prep"}} : memref<1x4x4xf32> to tensor<1x4x4xf32>
// IR: linalg.generic{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}
// IR: linalg.reduce{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @shared_load_chain(%lhs_memref: memref<*xf32>,
                             %rhs_a_memref: memref<*xf32>,
                             %rhs_b_memref: memref<*xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32

  %lhs_view = memref.reinterpret_cast %lhs_memref to offset: [0],
      sizes: [4, 1, 4], strides: [4, 4, 1]
      : memref<*xf32> to memref<4x1x4xf32, strided<[4, 4, 1]>>
  %lhs_buffer = memref.alloc() : memref<4x1x4xf32>
  memref.copy %lhs_view, %lhs_buffer
      : memref<4x1x4xf32, strided<[4, 4, 1]>> to memref<4x1x4xf32>
  %shared_lhs = bufferization.to_tensor %lhs_buffer restrict writable
      : memref<4x1x4xf32> to tensor<4x1x4xf32>

  %rhs_a_view = memref.reinterpret_cast %rhs_a_memref to offset: [0],
      sizes: [1, 4, 4], strides: [16, 4, 1]
      : memref<*xf32> to memref<1x4x4xf32, strided<[16, 4, 1]>>
  %rhs_a_buffer = memref.alloc() : memref<1x4x4xf32>
  memref.copy %rhs_a_view, %rhs_a_buffer
      : memref<1x4x4xf32, strided<[16, 4, 1]>> to memref<1x4x4xf32>
  %rhs_a = bufferization.to_tensor %rhs_a_buffer restrict writable
      : memref<1x4x4xf32> to tensor<1x4x4xf32>

  %product_a_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_a = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared_lhs, %rhs_a : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_a_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %final_a_empty = tensor.empty() : tensor<4x4xf32>
  %final_a_init = linalg.fill ins(%zero : f32)
      outs(%final_a_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_a = linalg.reduce ins(%product_a : tensor<4x4x4xf32>)
      outs(%final_a_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }

  %rhs_b_view = memref.reinterpret_cast %rhs_b_memref to offset: [0],
      sizes: [1, 4, 4], strides: [16, 4, 1]
      : memref<*xf32> to memref<1x4x4xf32, strided<[16, 4, 1]>>
  %rhs_b_buffer = memref.alloc() : memref<1x4x4xf32>
  memref.copy %rhs_b_view, %rhs_b_buffer
      : memref<1x4x4xf32, strided<[16, 4, 1]>> to memref<1x4x4xf32>
  %rhs_b = bufferization.to_tensor %rhs_b_buffer restrict writable
      : memref<1x4x4xf32> to tensor<1x4x4xf32>

  %product_b_empty = tensor.empty() : tensor<4x4x4xf32>
  %product_b = linalg.generic {
    indexing_maps = [#broadcast_m, #broadcast_n, #identity3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%shared_lhs, %rhs_b : tensor<4x1x4xf32>, tensor<1x4x4xf32>)
      outs(%product_b_empty : tensor<4x4x4xf32>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %out: f32):
    %value = arith.mulf %lhs_value, %rhs_value : f32
    linalg.yield %value : f32
  } -> tensor<4x4x4xf32>
  %final_b_empty = tensor.empty() : tensor<4x4xf32>
  %final_b_init = linalg.fill ins(%zero : f32)
      outs(%final_b_empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %final_b = linalg.reduce ins(%product_b : tensor<4x4x4xf32>)
      outs(%final_b_init : tensor<4x4xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %sum = arith.addf %final_a, %final_b : tensor<4x4xf32>
  return %sum : tensor<4x4xf32>
}
