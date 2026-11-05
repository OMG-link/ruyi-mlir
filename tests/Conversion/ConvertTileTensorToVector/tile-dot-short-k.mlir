// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -convert-tile-tensor-to-vector=allow-ime-tile-contracts -canonicalize -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 vlen=256" -convert-tile-tensor-to-vector=allow-ime-tile-contracts -canonicalize -split-input-file | FileCheck %s --check-prefix=K1
//
// A tile product whose K is below the tile's converts when both operands are
// loads a full-K tile can be read from (rows the tile's K apart, K
// consecutive) and nothing else reads those loads. The contract keeps its
// short K; widening it is a later pass's job. Anything else stays a
// linalg.matmul.

#id = affine_map<(i, j) -> (i, j)>

// The scale outer product: fp16 K = 1, rows 8 apart (the A100 fp16 tile's
// K), B transposed from an (N, 1) load.
// CHECK-LABEL: func.func @outer_product
// CHECK-DAG: %[[A:.+]] = vector.transfer_read {{.*}} : memref<8x1xf16, strided<[8, 1]>>, vector<8x1xf16>
// CHECK-DAG: %[[B:.+]] = vector.transfer_read {{.*}} : memref<8x1xf16, strided<[8, 1]>>, vector<8x1xf16>
// CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [1, 0]
// CHECK-DAG: %[[EA:.+]] = arith.extf %[[A]] : vector<8x1xf16> to vector<8x1xf32>
// CHECK-DAG: %[[EB:.+]] = arith.extf %[[BT]] : vector<1x8xf16> to vector<1x8xf32>
// CHECK: vector.contract {{.*}} %[[EA]], %[[EB]], {{.*}} : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
// CHECK-NOT: linalg.matmul
// The K1 has no fp16 tile.
// K1-LABEL: func.func @outer_product
// K1: linalg.matmul
func.func @outer_product(%a: memref<8x1xf16, strided<[8, 1]>>,
                         %b: memref<8x1xf16, strided<[8, 1]>>,
                         %c: memref<8x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x8xf32>
  %z = linalg.fill ins(%zero : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
  %ta = bufferization.to_tensor %a restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %tb = bufferization.to_tensor %b restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %eh = tensor.empty() : tensor<1x8xf16>
  %bt = linalg.transpose ins(%tb : tensor<8x1xf16>) outs(%eh : tensor<1x8xf16>) permutation = [1, 0]
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
  return
}

// -----

// B as a direct (1, N) load whose N elements lie the tile's K apart, and one
// extended A shared by two products: both convert.
// CHECK-LABEL: func.func @direct_b_shared_a
// CHECK-COUNT-2: vector.contract {{.*}} : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
// CHECK-NOT: linalg.matmul
#id = affine_map<(i, j) -> (i, j)>
func.func @direct_b_shared_a(%a: memref<8x1xf16, strided<[8, 1]>>,
                             %b: memref<1x8xf16, strided<[64, 8]>>,
                             %c: memref<8x8xf32>, %d: memref<8x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x8xf32>
  %z = linalg.fill ins(%zero : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
  %ta = bufferization.to_tensor %a restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %tb = bufferization.to_tensor %b restrict : memref<1x8xf16, strided<[64, 8]>> to tensor<1x8xf16>
  %ea0 = tensor.empty() : tensor<8x1xf32>
  %ea = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%ta : tensor<8x1xf16>) outs(%ea0 : tensor<8x1xf32>) {
  ^bb0(%x: f16, %y: f32):
    %v = arith.extf %x : f16 to f32
    linalg.yield %v : f32
  } -> tensor<8x1xf32>
  %eb0 = tensor.empty() : tensor<1x8xf32>
  %eb = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%tb : tensor<1x8xf16>) outs(%eb0 : tensor<1x8xf32>) {
  ^bb0(%x: f16, %y: f32):
    %v = arith.extf %x : f16 to f32
    linalg.yield %v : f32
  } -> tensor<1x8xf32>
  %m = linalg.matmul ins(%ea, %eb : tensor<8x1xf32>, tensor<1x8xf32>) outs(%z : tensor<8x8xf32>) -> tensor<8x8xf32>
  %n = linalg.matmul ins(%ea, %eb : tensor<8x1xf32>, tensor<1x8xf32>) outs(%m : tensor<8x8xf32>) -> tensor<8x8xf32>
  bufferization.materialize_in_destination %n in writable %c : (tensor<8x8xf32>, memref<8x8xf32>) -> ()
  return
}

// -----

// int8 with K = 4 below the tile's 16, rows 16 apart. On the K1 the int8
// tile is 4x4x8, so this M, N has no tile there.
// CHECK-LABEL: func.func @int8_short
// CHECK: vector.contract {{.*}} : vector<8x4xi32>, vector<4x8xi32> into vector<8x8xi32>
// CHECK-NOT: linalg.matmul
// K1-LABEL: func.func @int8_short
// K1: linalg.matmul
func.func @int8_short(%a: memref<8x4xi8, strided<[16, 1]>>,
                      %b: memref<8x4xi8, strided<[16, 1]>>,
                      %c: memref<8x8xi32>) {
  %zero = arith.constant 0 : i32
  %e = tensor.empty() : tensor<8x8xi32>
  %z = linalg.fill ins(%zero : i32) outs(%e : tensor<8x8xi32>) -> tensor<8x8xi32>
  %ta = bufferization.to_tensor %a restrict : memref<8x4xi8, strided<[16, 1]>> to tensor<8x4xi8>
  %tb = bufferization.to_tensor %b restrict : memref<8x4xi8, strided<[16, 1]>> to tensor<8x4xi8>
  %eh = tensor.empty() : tensor<4x8xi8>
  %bt = linalg.transpose ins(%tb : tensor<8x4xi8>) outs(%eh : tensor<4x8xi8>) permutation = [1, 0]
  %m = linalg.matmul ins(%ta, %bt : tensor<8x4xi8>, tensor<4x8xi8>) outs(%z : tensor<8x8xi32>) -> tensor<8x8xi32>
  bufferization.materialize_in_destination %m in writable %c : (tensor<8x8xi32>, memref<8x8xi32>) -> ()
  return
}

// -----

// Rows 16 apart cannot be read as an 8-wide fp16 K.
// CHECK-LABEL: func.func @wrong_row_stride
// CHECK: linalg.matmul
// CHECK-NOT: vector.contract
#id = affine_map<(i, j) -> (i, j)>
func.func @wrong_row_stride(%a: memref<8x1xf16, strided<[16, 1]>>,
                            %b: memref<8x1xf16, strided<[8, 1]>>,
                            %c: memref<8x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x8xf32>
  %z = linalg.fill ins(%zero : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
  %ta = bufferization.to_tensor %a restrict : memref<8x1xf16, strided<[16, 1]>> to tensor<8x1xf16>
  %tb = bufferization.to_tensor %b restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %eh = tensor.empty() : tensor<1x8xf16>
  %bt = linalg.transpose ins(%tb : tensor<8x1xf16>) outs(%eh : tensor<1x8xf16>) permutation = [1, 0]
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
  return
}

// -----

// The A load has another reader, which widening would change.
// CHECK-LABEL: func.func @other_reader
// CHECK: linalg.matmul
// CHECK-NOT: vector.contract
func.func @other_reader(%a: memref<8x4xi8, strided<[16, 1]>>,
                        %b: memref<8x4xi8, strided<[16, 1]>>,
                        %c: memref<8x8xi32>, %d: memref<8x4xi8>) {
  %zero = arith.constant 0 : i32
  %e = tensor.empty() : tensor<8x8xi32>
  %z = linalg.fill ins(%zero : i32) outs(%e : tensor<8x8xi32>) -> tensor<8x8xi32>
  %ta = bufferization.to_tensor %a restrict : memref<8x4xi8, strided<[16, 1]>> to tensor<8x4xi8>
  %tb = bufferization.to_tensor %b restrict : memref<8x4xi8, strided<[16, 1]>> to tensor<8x4xi8>
  %eh = tensor.empty() : tensor<4x8xi8>
  %bt = linalg.transpose ins(%tb : tensor<8x4xi8>) outs(%eh : tensor<4x8xi8>) permutation = [1, 0]
  %m = linalg.matmul ins(%ta, %bt : tensor<8x4xi8>, tensor<4x8xi8>) outs(%z : tensor<8x8xi32>) -> tensor<8x8xi32>
  bufferization.materialize_in_destination %m in writable %c : (tensor<8x8xi32>, memref<8x8xi32>) -> ()
  bufferization.materialize_in_destination %ta in writable %d : (tensor<8x4xi8>, memref<8x4xi8>) -> ()
  return
}

// -----

// A full-K product still needs B's transpose: a direct K x N B has no tile
// dot form.
// CHECK-LABEL: func.func @full_k_direct_b
// CHECK: linalg.matmul
// CHECK-NOT: vector.contract
func.func @full_k_direct_b(%a: memref<8x16xi8>, %b: memref<16x8xi8>,
                           %c: memref<8x8xi32>) {
  %zero = arith.constant 0 : i32
  %e = tensor.empty() : tensor<8x8xi32>
  %z = linalg.fill ins(%zero : i32) outs(%e : tensor<8x8xi32>) -> tensor<8x8xi32>
  %ta = bufferization.to_tensor %a restrict : memref<8x16xi8> to tensor<8x16xi8>
  %tb = bufferization.to_tensor %b restrict : memref<16x8xi8> to tensor<16x8xi8>
  %m = linalg.matmul ins(%ta, %tb : tensor<8x16xi8>, tensor<16x8xi8>) outs(%z : tensor<8x8xi32>) -> tensor<8x8xi32>
  bufferization.materialize_in_destination %m in writable %c : (tensor<8x8xi32>, memref<8x8xi32>) -> ()
  return
}
