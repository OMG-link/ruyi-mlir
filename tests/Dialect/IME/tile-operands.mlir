// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -test-ime-tile-operands -o /dev/null | FileCheck %s
//
// The operand analysis shared by the IME tile release and K promotion: each
// product is matched to an A100 tile by M and N, and each operand is traced
// through zero-preserving casts (and B's transpose) to its load, which is
// checked for widening to the tile's full K.

#id = affine_map<(i, j) -> (i, j)>

// Scale outer product: K = 1, rows 8 apart (the fp16 tile's K), B as the
// transpose of an (N, 1) load. Both operands can be widened.
// CHECK-LABEL: @outer_product linalg.matmul 8x8x1 float: tile k=8
// CHECK-NEXT: A: bufferization.to_tensor memref<8x1xf16, strided<[8, 1]>> row=0 k=1 transposed=0 chain=1 widen=1
// CHECK-NEXT: B: bufferization.to_tensor memref<8x1xf16, strided<[8, 1]>> row=0 k=1 transposed=1 chain=2 widen=1
func.func @outer_product(%a: memref<8x1xf16, strided<[8, 1]>>,
                         %b: memref<8x1xf16, strided<[8, 1]>>) -> tensor<8x8xf32> {
  %ta = bufferization.to_tensor %a restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %tb = bufferization.to_tensor %b restrict : memref<8x1xf16, strided<[8, 1]>> to tensor<8x1xf16>
  %ea0 = tensor.empty() : tensor<8x1xf32>
  %ea = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%ta : tensor<8x1xf16>) outs(%ea0 : tensor<8x1xf32>) {
  ^bb0(%x: f16, %y: f32):
    %e = arith.extf %x : f16 to f32
    linalg.yield %e : f32
  } -> tensor<8x1xf32>
  %eb = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%tb : tensor<8x1xf16>) outs(%ea0 : tensor<8x1xf32>) {
  ^bb0(%x: f16, %y: f32):
    %e = arith.extf %x : f16 to f32
    linalg.yield %e : f32
  } -> tensor<8x1xf32>
  %t0 = tensor.empty() : tensor<1x8xf32>
  %bt = linalg.transpose ins(%eb : tensor<8x1xf32>) outs(%t0 : tensor<1x8xf32>) permutation = [1, 0]
  %c0 = tensor.empty() : tensor<8x8xf32>
  %r = linalg.matmul ins(%ea, %bt : tensor<8x1xf32>, tensor<1x8xf32>) outs(%c0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// Rows 16 apart cannot be read as an 8-wide K; a direct (K, N) B reads its
// rows along N, which are not 8 apart.
// CHECK-LABEL: @bad_strides linalg.matmul 8x8x1 float: tile k=8
// CHECK-NEXT: A: bufferization.to_tensor memref<8x1xf16, strided<[16, 1]>> row=0 k=1 transposed=0 chain=1 widen=0
// CHECK-NEXT: B: bufferization.to_tensor memref<1x8xf16> row=1 k=0 transposed=0 chain=1 widen=0
func.func @bad_strides(%a: memref<8x1xf16, strided<[16, 1]>>,
                       %b: memref<1x8xf16>) -> tensor<8x8xf32> {
  %ta = bufferization.to_tensor %a restrict : memref<8x1xf16, strided<[16, 1]>> to tensor<8x1xf16>
  %tb = bufferization.to_tensor %b restrict : memref<1x8xf16> to tensor<1x8xf16>
  %ea0 = tensor.empty() : tensor<8x1xf32>
  %ea = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%ta : tensor<8x1xf16>) outs(%ea0 : tensor<8x1xf32>) {
  ^bb0(%x: f16, %y: f32):
    %e = arith.extf %x : f16 to f32
    linalg.yield %e : f32
  } -> tensor<8x1xf32>
  %eb0 = tensor.empty() : tensor<1x8xf32>
  %eb = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%tb : tensor<1x8xf16>) outs(%eb0 : tensor<1x8xf32>) {
  ^bb0(%x: f16, %y: f32):
    %e = arith.extf %x : f16 to f32
    linalg.yield %e : f32
  } -> tensor<1x8xf32>
  %c0 = tensor.empty() : tensor<8x8xf32>
  %r = linalg.matmul ins(%ea, %eb : tensor<8x1xf32>, tensor<1x8xf32>) outs(%c0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// A generic that is not a single cast stops the trace.
// CHECK-LABEL: @not_a_cast linalg.matmul 8x8x1 float: tile k=8
// CHECK-NEXT: A: untraced
func.func @not_a_cast(%a: memref<8x1xf32>, %b: tensor<1x8xf32>) -> tensor<8x8xf32> {
  %ta = bufferization.to_tensor %a restrict : memref<8x1xf32> to tensor<8x1xf32>
  %ea0 = tensor.empty() : tensor<8x1xf32>
  %ea = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%ta : tensor<8x1xf32>) outs(%ea0 : tensor<8x1xf32>) {
  ^bb0(%x: f32, %y: f32):
    %e = arith.addf %x, %x : f32
    linalg.yield %e : f32
  } -> tensor<8x1xf32>
  %c0 = tensor.empty() : tensor<8x8xf32>
  %r = linalg.matmul ins(%ea, %b : tensor<8x1xf32>, tensor<1x8xf32>) outs(%c0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// Vector level: the full int8 tile, B as a transposed (N, K) read.
// CHECK-LABEL: @vector_tile vector.contract 8x8x16 int: tile k=16
// CHECK-NEXT: A: vector.transfer_read memref<8x16xi8> row=0 k=1 transposed=0 chain=1 widen=1
// CHECK-NEXT: B: vector.transfer_read memref<8x16xi8> row=0 k=1 transposed=1 chain=2 widen=1
func.func @vector_tile(%a: memref<8x16xi8>, %b: memref<8x16xi8>,
                       %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i8
  %va = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x16xi8>, vector<8x16xi8>
  %vb = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x16xi8>, vector<8x16xi8>
  %ea = arith.extsi %va : vector<8x16xi8> to vector<8x16xi32>
  %eb = arith.extsi %vb : vector<8x16xi8> to vector<8x16xi32>
  %bt = vector.transpose %eb, [1, 0] : vector<8x16xi32> to vector<16x8xi32>
  %r = vector.contract {indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                                         affine_map<(m, n, k) -> (k, n)>,
                                         affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %ea, %bt, %acc
      : vector<8x16xi32>, vector<16x8xi32> into vector<8x8xi32>
  return %r : vector<8x8xi32>
}

// No A100 tile has this M and N.
// CHECK-LABEL: @no_tile linalg.matmul 3x5x4 int: no tile
// CHECK-NEXT: A: untraced
// CHECK-NEXT: B: untraced
func.func @no_tile(%a: tensor<3x4xi8>, %b: tensor<4x5xi8>) -> tensor<3x5xi32> {
  %c0 = tensor.empty() : tensor<3x5xi32>
  %r = linalg.matmul ins(%a, %b : tensor<3x4xi8>, tensor<4x5xi8>) outs(%c0 : tensor<3x5xi32>) -> tensor<3x5xi32>
  return %r : tensor<3x5xi32>
}
