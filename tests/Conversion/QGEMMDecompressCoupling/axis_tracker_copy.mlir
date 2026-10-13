// RUN: bash %S/extract_copied_helpers.sh %S/../../../midend/lib/Conversion/QGEMMRecognize/Recognition.cpp > %t.recognizer
// RUN: bash %S/extract_copied_helpers.sh %S/../../../midend/lib/Conversion/QGEMMDecompressCoupling/AxisTracker.cpp %S/../../../midend/lib/Conversion/QGEMMDecompressCoupling/AxisTracker.h > %t.copy
// RUN: diff %t.recognizer %t.copy
// RUN: bash %S/extract_copied_helpers.sh --list-region %S/../../../midend/lib/Conversion/QGEMMDecompressCoupling/AxisTracker.cpp > %t.region
// RUN: bash %S/extract_copied_helpers.sh --list-expected > %t.expected
// RUN: test -s %t.region
// RUN: diff %t.expected %t.region
//
// Guards the verbatim copy of the recognizer's dimension-mapping helpers
// (DimMapping, getRank, makeIdentityMapping, makeNullMapping, resizeMapping,
// mapThroughAffineMap, getIterationMapping, getProducerMapping) inside
// AxisTracker.cpp's "Begin copy"/"End copy" region. Any drift fails the diff.
// The set of static functions defined in the region must equal the compared
// list, so a helper added to the copy without being compared also fails.
