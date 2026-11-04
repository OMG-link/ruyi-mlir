// RUN: buddy-opt %s -lower-ime="target=a100" | FileCheck %s --check-prefix=MLIR
// RUN: buddy-opt %s -lower-ime="target=a100" -convert-vector-to-llvm -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN:   | buddy-translate -buddy-to-llvmir \
// RUN:   | buddy-llc -mtriple=riscv64 -mattr=+m,+v,+zvfh,+zvl1024b,+xsmtvdotii -verify-machineinstrs \
// RUN:   | FileCheck %s --check-prefix=ASM
// RUN: buddy-opt %s -lower-ime -split-input-file -verify-diagnostics -o /dev/null
//
// Value-level A100 tile dots (VLEN 1024): an int8 8x16 tile and an fp16 8x8
// tile are one register each, and the 8x8 int32 or fp32 accumulator a
// register pair. Each lowers to one dot on the registers, with no load or
// store around it. The K1 lowering (VLEN 256) rejects these shapes.

// MLIR-LABEL: func.func @tile_vmadot
// MLIR-DAG: llvm.intr.vector.insert {{.*}} : vector<64xi32> into vector<[4]xi32>
// MLIR-DAG: llvm.intr.vector.insert {{.*}} : vector<128xi8> into vector<[8]xi8>
// MLIR: ime.intr.vmadot
// ASM-LABEL: tile_vmadot:
// ASM-NOT: vle
// ASM: smt.vmadot
// ASM-NOT: vse
// ASM: ret
func.func @tile_vmadot(%c: vector<64xi32>, %a: vector<128xi8>, %b: vector<128xi8>) -> vector<64xi32> {
  // expected-error @+2 {{operands are not one register tile and one register-pair accumulator at VLEN 256}}
  // expected-error @+1 {{failed to legalize operation 'ime.tile.vmadot'}}
  %r = ime.tile.vmadot %c, %a, %b : vector<128xi8>, vector<128xi8> into vector<64xi32>
  return %r : vector<64xi32>
}

// -----

// MLIR-LABEL: func.func @tile_vfwmadot
// MLIR-DAG: llvm.intr.vector.insert {{.*}} : vector<64xf32> into vector<[4]xf32>
// MLIR-DAG: llvm.intr.vector.insert {{.*}} : vector<64xf16> into vector<[4]xf16>
// MLIR: ime.intr.vfmadot
// ASM-LABEL: tile_vfwmadot:
// ASM-NOT: vle
// ASM: smt.vfwmadot
// ASM-NOT: vse
// ASM: ret
func.func @tile_vfwmadot(%c: vector<64xf32>, %a: vector<64xf16>, %b: vector<64xf16>) -> vector<64xf32> {
  // expected-error @+2 {{has no instruction on the SpacemiT K1 (XSMTVDot)}}
  // expected-error @+1 {{failed to legalize operation 'ime.tile.vfwmadot'}}
  %r = ime.tile.vfwmadot %c, %a, %b : vector<64xf16>, vector<64xf16> into vector<64xf32>
  return %r : vector<64xf32>
}
