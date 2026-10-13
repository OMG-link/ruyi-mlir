//===- AxisTracker.h - Axis tracking for QGEMM coupling ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 (the "License");
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
//
// The axis tracker answers, for every value of Omega(r), at which position of
// its type the parallel dimensions m and n of the final reduction sit.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_QGEMM_DECOMPRESS_COUPLING_AXIS_TRACKER_H
#define BUDDY_QGEMM_DECOMPRESS_COUPLING_AXIS_TRACKER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace buddy {
namespace qgemm_coupling {

/// Maps each dimension of a value to a dimension of the final reduction's
/// result, or to none. Same definition as the recognizer's.
using DimMapping = SmallVector<std::optional<unsigned>>;

/// The two parallel axes of a QGEMM root.
enum class Axis : unsigned { M = 0, N = 1 };

/// Consumer-ward mapping: given the DimMapping of `operand` of `consumer`,
/// return the mapping of the consumer's first result. Generic indexing maps
/// are inverted over their projected-permutation part; linalg.reduce
/// `dimensions`, linalg.transpose permutations and reshape reassociation
/// groups are inverted. Root dims that land in a reassociation group of size
/// > 1 are appended to `groupedRootDims` when it is non-null. Returns an
/// empty mapping when the consumer has no ranked shaped result.
DimMapping
getResultMappingFromOperand(Operation *consumer, OpOperand &operand,
                            const DimMapping &operandMapping,
                            SmallVectorImpl<unsigned> *groupedRootDims);

/// Axis tracker for one QGEMM root: producer-ward copies of the recognizer
/// plus consumer-ward propagation.
class AxisTracker {
public:
  /// `omega` is the root's op set; `mDim` / `nDim` come from qgemm.root.
  AxisTracker(Operation *finalReduce, const SetVector<Operation *> &omega,
              unsigned mDim, unsigned nDim);

  /// Walk producer-ward from the final reduction, seeded with the identity
  /// mapping of its result.
  void run();

  /// Fill null entries reachable only consumer-ward. Iterates to a fixed
  /// point over Omega(r); called by run().
  void propagateToUsers();
  /// Position at which the user of `use` expects `axis` in its operand type.
  /// Used for Outside_Users (Req 6.5): their operand type is fixed, so the
  /// position is the one the tracked value carries.
  std::optional<unsigned> getExpectedPosition(OpOperand &use, Axis axis) const;
  /// True when `op` is a reshape whose reassociation groups `axis` with
  /// another dimension (Req 7.6).
  bool isGrouped(Operation *op, Axis axis) const {
    return grouped.contains({op, static_cast<unsigned>(axis)});
  }
  /// Mapping of a value reached by the walk, or nullptr if never reached.
  const DimMapping *getMapping(Value value) const;

  /// Position of axis `axis` in `value`'s type, if the walk assigned one.
  std::optional<unsigned> getAxisPosition(Value value, Axis axis) const;

  unsigned getRootDim(Axis axis) const { return axis == Axis::M ? mDim : nDim; }

  Operation *getFinalReduce() const { return finalReduce; }

  void print(llvm::raw_ostream &os) const;

private:
  void visitOperation(Operation *operation, const DimMapping &iterationMapping);
  /// Carry a memref.copy target's mapping to its source.
  void visitCopySources();
  /// Returns true when the stored mapping changed.
  bool recordValue(Value value, const DimMapping &mapping);
  void markGrouped();
  /// Static extent of `rootDim` in the final reduction's result, if static.
  std::optional<int64_t> getRootExtent(unsigned rootDim) const;
  /// Keep one position per root dim, preferring the one whose static extent
  /// matches the root dim's.
  DimMapping disambiguate(Value value, const DimMapping &mapping) const;
  /// Assert that every tracked axis position has the root dim's extent.
  void verifyExtents() const;

  Operation *finalReduce;
  const SetVector<Operation *> &omega;
  unsigned mDim;
  unsigned nDim;
  DenseMap<Value, DimMapping> valueMappings;
  SmallVector<Value> valueOrder;
  DenseSet<Operation *> visited;
  DenseSet<std::pair<Operation *, unsigned>> grouped;
};

} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir

#endif // BUDDY_QGEMM_DECOMPRESS_COUPLING_AXIS_TRACKER_H
