// RUN: buddy-opt %s -lower-ime | FileCheck %s --check-prefix=MLIR
// RUN: buddy-opt %s -lower-ime -convert-vector-to-llvm -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN:   | buddy-translate -buddy-to-llvmir \
// RUN:   | buddy-llc -mtriple=riscv64 -mattr=+m,+v,+zvl256b,+xsmtvdot -verify-machineinstrs \
// RUN:   | FileCheck %s --check-prefix=ASM
// RUN: buddy-opt %s -lower-ime="target=k3" -verify-diagnostics -split-input-file -o /dev/null
//
// Value-level K1 tile dots: each fixed tile is placed in the low part of its
// scalable register type, so at VLEN = 256 the dot reads and writes
// registers directly, with no load or store around it.

// MLIR-LABEL: func.func @tile_vmadot
// MLIR-DAG: llvm.intr.vector.insert {{.*}} : vector<16xi32> into vector<[4]xi32>
// MLIR-DAG: llvm.intr.vector.insert {{.*}} : vector<32xi8> into vector<[8]xi8>
// MLIR: ime.intr.vmadot
// MLIR: llvm.intr.vector.extract {{.*}} : vector<16xi32> from vector<[4]xi32>
// MLIR-NOT: ime.tile
// ASM-LABEL: tile_vmadot:
// ASM-NOT: vle
// ASM: smt.vmadot
// ASM-NOT: vse
// ASM: ret
func.func @tile_vmadot(%c: vector<16xi32>, %a: vector<32xi8>, %b: vector<32xi8>) -> vector<16xi32> {
  // expected-error @+2 {{operands are not one register tile and one register-pair accumulator at VLEN 1024}}
  // expected-error @+1 {{failed to legalize operation 'ime.tile.vmadot'}}
  %r = ime.tile.vmadot %c, %a, %b : vector<32xi8>, vector<32xi8> into vector<16xi32>
  return %r : vector<16xi32>
}

// -----

// MLIR-LABEL: func.func @tile_vmadotu
// MLIR: ime.intr.vmadotu
// ASM-LABEL: tile_vmadotu:
// ASM: smt.vmadotu
func.func @tile_vmadotu(%c: vector<16xi32>, %a: vector<32xi8>, %b: vector<32xi8>) -> vector<16xi32> {
  // expected-error @+2 {{operands are not one register tile and one register-pair accumulator at VLEN 1024}}
  // expected-error @+1 {{failed to legalize operation 'ime.tile.vmadotu'}}
  %r = ime.tile.vmadotu %c, %a, %b : vector<32xi8>, vector<32xi8> into vector<16xi32>
  return %r : vector<16xi32>
}
