//===-- CParDialect.td - CPU Parallel Region Dialect -------*- tablegen -*-===//
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

#ifndef CPAR_CPARDIALECT_TD
#define CPAR_CPARDIALECT_TD

include "mlir/IR/OpBase.td"

def CPar_Dialect : Dialect {
  let name = "cpar";
  let summary = "CPU parallel regions executed by a fork/join runtime";
  let description = [{
    The `cpar` dialect describes code regions that are executed by several CPU
    threads at the same time (fork/join, SPMD).  A region stays a regular
    nested region through the whole lowering pipeline, so the ops inside it
    are lowered by the usual conversion passes.  Once everything is in the
    LLVM dialect, `--cpar-outline-to-runtime` outlines every region into a
    function and replaces it with one call to the runtime:

    ```c
    void __cpar_parallel(void (*fn)(void *ctx, int64_t tid, int64_t nthreads),
                         void *ctx, int64_t nthreads);
    ```

    The runtime calls `fn(ctx, tid, nthreads)` once for every `tid` in
    `[0, nthreads)`, possibly from several threads, and returns when all calls
    have finished.  `ctx` points to the values the region uses from the
    enclosing code; they are shared by all threads (no copy of the data they
    point to is made).
  }];
  let cppNamespace = "::buddy::cpar";
}

#endif // CPAR_CPARDIALECT_TD
