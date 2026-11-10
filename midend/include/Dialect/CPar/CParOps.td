//===-- CParOps.td - CPU Parallel Region Operations --------*- tablegen -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef CPAR_CPAROPS_TD
#define CPAR_CPAROPS_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "CPar/CParDialect.td"

class CPar_Op<string mnemonic, list<Trait> traits = []> :
    Op<CPar_Dialect, mnemonic, traits>;

def CPar_ParallelOp : CPar_Op<"parallel", [
    AutomaticAllocationScope, AffineScope, RecursiveMemoryEffects]> {
  let summary = "Region executed by `num_threads` threads (fork/join)";
  let description = [{
    The body is executed once for every logical thread id
    `%tid` in `[0, num_threads)`; `%nthreads` is `num_threads`.  The executions
    may run concurrently on different CPU threads, the op completes when all
    of them have completed.  Values defined above the region are shared by all
    executions; values defined inside the region are private to one execution.

    The region is SPMD: dividing the work between the threads is up to the
    body, typically by deriving loop bounds from `%tid` and `%nthreads` (see
    `--cpar-distribute-loops`).

    The region is an automatic allocation scope: stack allocations made inside
    it are private to one execution.  The region may have several blocks
    (e.g. after `--convert-scf-to-cf`); blocks that leave the region end with
    `cpar.yield`.

    ```mlir
    cpar.parallel num_threads(4) {
    ^bb0(%tid: i64, %nthreads: i64):
      ...
      cpar.yield
    }
    ```
  }];

  let arguments = (ins ConfinedAttr<I64Attr, [IntPositive]>:$num_threads);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "`num_threads` `(` $num_threads `)` attr-dict-with-keyword $body";
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    // Creates the op with an entry block (%tid, %nthreads).  The body builder
    // is called with the insertion point inside the entry block; it must not
    // create the terminator, cpar.yield is appended afterwards.
    OpBuilder<(ins "int64_t":$numThreads,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
           "mlir::Value, mlir::Value)>", "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    mlir::Value getThreadId() { return getBody().front().getArgument(0); }
    mlir::Value getNumThreadsValue() {
      return getBody().front().getArgument(1);
    }
  }];
}

def CPar_YieldOp : CPar_Op<"yield", [
    Pure, Terminator, HasParent<"ParallelOp">]> {
  let summary = "Leaves a cpar.parallel region";
  let assemblyFormat = "attr-dict";
}

#endif // CPAR_CPAROPS_TD
