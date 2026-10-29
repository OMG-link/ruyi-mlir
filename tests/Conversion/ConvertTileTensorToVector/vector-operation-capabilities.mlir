// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=haswell" -convert-tile-tensor-to-vector -split-input-file | FileCheck %s --check-prefix=F16C
// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=x86-64" -convert-tile-tensor-to-vector -split-input-file | FileCheck %s --check-prefix=BASE
// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=icelake-server" -convert-tile-tensor-to-vector -split-input-file | FileCheck %s --check-prefix=AVX512
// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=sapphirerapids" -convert-tile-tensor-to-vector -split-input-file | FileCheck %s --check-prefix=FP16
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 features=+zvfhmin,-zvfh vlen=256" -convert-tile-tensor-to-vector -split-input-file | FileCheck %s --check-prefix=ZVFHMIN
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 vlen=256" -convert-tile-tensor-to-vector -split-input-file | FileCheck %s --check-prefix=ZVFH
// RUN: buddy-opt %s -set-llvm-target="triple=riscv64-unknown-linux-gnu chip=generic features=+zve32x,+zvl256b" -convert-tile-tensor-to-vector -split-input-file | FileCheck %s --check-prefix=NOFP

// F16C admits f16<->f32 conversion when input/output types differ.
// F16C-LABEL: func.func @f16_convert
// F16C: arith.extf {{.*}} : vector<16xf16> to vector<16xf32>
// F16C: vector.transfer_write
// BASE-LABEL: func.func @f16_convert
// BASE: linalg.generic
func.func @f16_convert(%src: memref<16xf16>, %dst: memref<16xf32>) {
  %x = bufferization.to_tensor %src restrict : memref<16xf16> to tensor<16xf16>
  %e = tensor.empty() : tensor<16xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%x : tensor<16xf16>) outs(%e : tensor<16xf32>) {
    ^bb0(%a: f16, %o: f32):
      %r = arith.extf %a : f16 to f32
      linalg.yield %r : f32
  } -> tensor<16xf32>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// f16 storage alone does not admit general f16 arithmetic on x86-64.
// BASE-LABEL: func.func @f16_add
// BASE: arith.addf {{.*}} : tensor<16xf16>
// F16C-LABEL: func.func @f16_add
// F16C: arith.addf {{.*}} : tensor<16xf16>
// ZVFHMIN-LABEL: func.func @f16_add
// ZVFHMIN: arith.addf
// ZVFHMIN-NOT: vector.addf
// FP16-LABEL: func.func @f16_add
// FP16: arith.addf {{.*}} : vector<16xf16>
// ZVFH-LABEL: func.func @f16_add
// ZVFH: arith.addf {{.*}} : vector<16xf16>
func.func @f16_add(%src: memref<16xf16>, %dst: memref<16xf16>) {
  %x = bufferization.to_tensor %src restrict : memref<16xf16> to tensor<16xf16>
  %y = arith.addf %x, %x : tensor<16xf16>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<16xf16>, memref<16xf16>) -> ()
  return
}

// -----

// The direct tensor elementwise path accepts the same f16/f32 conversion.
// F16C-LABEL: func.func @direct_convert
// F16C: arith.truncf {{.*}} : vector<16xf32> to vector<16xf16>
// F16C: vector.transfer_write
// BASE-LABEL: func.func @direct_convert
// BASE: arith.truncf {{.*}} : tensor<16xf32> to tensor<16xf16>
func.func @direct_convert(%src: memref<16xf32>, %dst: memref<16xf16>) {
  %x = bufferization.to_tensor %src restrict : memref<16xf32> to tensor<16xf32>
  %y = arith.truncf %x : tensor<16xf32> to tensor<16xf16>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<16xf16>, memref<16xf16>) -> ()
  return
}

// -----

// Compare masks are allowed as intermediate generic values and feed select.
// BASE-LABEL: func.func @compare_select
// BASE: arith.cmpf
// BASE: arith.select
// BASE: vector.transfer_write
func.func @compare_select(%lhs: memref<16xf32>, %rhs: memref<16xf32>, %dst: memref<16xf32>) {
  %a = bufferization.to_tensor %lhs restrict : memref<16xf32> to tensor<16xf32>
  %b = bufferization.to_tensor %rhs restrict : memref<16xf32> to tensor<16xf32>
  %e = tensor.empty() : tensor<16xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) {
    ^bb0(%x: f32, %z: f32, %o: f32):
      %cmp = arith.cmpf ogt, %x, %z : f32
      %v = arith.select %cmp, %x, %z : f32
      linalg.yield %v : f32
  } -> tensor<16xf32>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// Unknown arithmetic must remain in tensor form despite supported storage.
// BASE-LABEL: func.func @unknown_integer_op
// BASE: arith.remui {{.*}} : tensor<16xi32>
func.func @unknown_integer_op(%src: memref<16xi32>, %dst: memref<16xi32>) {
  %x = bufferization.to_tensor %src restrict : memref<16xi32> to tensor<16xi32>
  %y = arith.remui %x, %x : tensor<16xi32>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<16xi32>, memref<16xi32>) -> ()
  return
}

// -----

// Integer-only RVV permits f32 storage, but not f32 arithmetic.
// NOFP-LABEL: func.func @float_add
// NOFP: arith.addf {{.*}} : tensor<16xf32>
func.func @float_add(%src: memref<16xf32>, %dst: memref<16xf32>) {
  %x = bufferization.to_tensor %src restrict : memref<16xf32> to tensor<16xf32>
  %y = arith.addf %x, %x : tensor<16xf32>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// AVX-512F also has vector f16/f32 conversion forms, independent of F16C and
// AVX512FP16. General f16 arithmetic remains unavailable.
// AVX512-LABEL: func.func @f16_convert_avx512
// AVX512: arith.truncf {{.*}} : vector<16xf32> to vector<16xf16>
// AVX512: vector.transfer_write
func.func @f16_convert_avx512(%src: memref<16xf32>, %dst: memref<16xf16>) {
  %x = bufferization.to_tensor %src restrict : memref<16xf32> to tensor<16xf32>
  %e = tensor.empty() : tensor<16xf16>
  %y = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%x : tensor<16xf32>) outs(%e : tensor<16xf16>) {
    ^bb0(%a: f32, %o: f16):
      %r = arith.truncf %a : f32 to f16
      linalg.yield %r : f16
  } -> tensor<16xf16>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<16xf16>, memref<16xf16>) -> ()
  return
}

// -----

// Zvfhmin permits f16<->f32 conversion without enabling f16 arithmetic.
// ZVFHMIN-LABEL: func.func @f16_convert_rvv
// ZVFHMIN: arith.extf {{.*}} : vector<16xf16> to vector<16xf32>
func.func @f16_convert_rvv(%src: memref<16xf16>, %dst: memref<16xf32>) {
  %x = bufferization.to_tensor %src restrict : memref<16xf16> to tensor<16xf16>
  %e = tensor.empty() : tensor<16xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%x : tensor<16xf16>) outs(%e : tensor<16xf32>) {
    ^bb0(%a: f16, %o: f32):
      %r = arith.extf %a : f16 to f32
      linalg.yield %r : f32
  } -> tensor<16xf32>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}
