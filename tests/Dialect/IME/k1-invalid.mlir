// RUN: buddy-opt %s -split-input-file -lower-ime -verify-diagnostics
//
// The SpacemiT K1 (XSMTVDot) has no floating-point dot, no dynamic slide and
// no int16 dot; the generic target rejects them instead of emitting
// instructions the core cannot decode.

func.func @k1_fp16(%c: memref<4x4xf16>, %a: memref<4x4xf16>, %b: memref<4x4xf16>) {
  // expected-error @below {{has no instruction on the SpacemiT K1}}
  // expected-error @below {{failed to legalize}}
  ime.vfmadot %c, %a, %b : memref<4x4xf16>, memref<4x4xf16>, memref<4x4xf16>
  return
}

// -----

func.func @k1_dynamic_slide(%c: memref<4x4xi32>, %a: memref<8x8xi8>, %b: memref<4x8xi8>, %n: i64) {
  // expected-error @below {{has no instruction on the SpacemiT K1}}
  // expected-error @below {{failed to legalize}}
  ime.vmadotn %c, %a, %b, %n : memref<4x4xi32>, memref<8x8xi8>, memref<4x8xi8>
  return
}

// -----

func.func @k1_int16(%c: memref<4x4xi32>, %a: memref<4x4xi16>, %b: memref<4x4xi16>) {
  // expected-error @below {{supports only int8 inputs}}
  // expected-error @below {{failed to legalize}}
  ime.vmadot %c, %a, %b : memref<4x4xi32>, memref<4x4xi16>, memref<4x4xi16>
  return
}
