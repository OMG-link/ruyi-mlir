// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q80_q80_strict.ttshared.mlir 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict.ttshared.mlir 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %S/Inputs/q80_q80_strict.ttshared.mlir 2>&1 >/dev/null | FileCheck %s --check-prefix=PRE --allow-empty
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %S/Inputs/q80_q80_strict.ttshared.mlir 2>&1 >/dev/null | FileCheck %s --check-prefix=OTF
//
// Req 14.7, 14.8, 12.22: the unmarked capture carries no `llvm.noalias`, and
// it writes its output argument %arg4 through
// bufferization.materialize_in_destination. Every load chain's base may
// therefore alias a written argument (W4), so Load_Copy_Forwarding keeps
// every alloc and copy under both modes. The `_noalias` copy of the same
// capture is what the forwarding tests use.
//
// All four load chains keep their alloc, copy and to_tensor of the alloc.
// IR-COUNT-4: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} {qgemm.node
// IR-NOT: memref.copy
// IR: bufferization.materialize_in_destination
//
// Under pre a kept copy changes nothing and nothing is reported.
// PRE-NOT: remark
//
// Under on-the-fly each kept copy is a Default op that cannot take the
// missing parallel dim, so it gets exactly one Remark R3, and so does the
// to_tensor of its alloc (only a memref.reinterpret_cast has an insert rule).
// No stride-0 view is emitted.
// OTF-COUNT-4: note: see current operation: %{{[a-z0-9_]+}} = bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable
// OTF-COUNT-4: note: see current operation: memref.copy
// OTF-NOT: remark
