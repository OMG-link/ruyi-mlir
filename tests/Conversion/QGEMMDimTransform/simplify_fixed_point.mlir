// This lit test replaces the former external hash sweep with executable
// determinism and idempotence checks for the dimension-transform simplifier.
// Each input is run twice to compare the bytes produced by separate runs, and
// the first output is fed back to the pass to compare the bytes at its fixed
// point. The same first output is also canonicalized by mlir-opt; the total
// expand/collapse/transpose count must remain unchanged.
//
// The count assertions below are deliberately total-count constraints over the
// complete output. They are not equivalent to per-tensor producer-chain counts:
// they do not prove each individual tensor has the prescribed producer chain.

// q80_q80_strict: expand/collapse/transpose = 2/0/0.
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir > %t.q80_q80_strict.first
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q80_q80_strict.ttshared.mlir > %t.q80_q80_strict.second
// RUN: diff -u %t.q80_q80_strict.first %t.q80_q80_strict.second
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %t.q80_q80_strict.first > %t.q80_q80_strict.idempotent
// RUN: diff -u %t.q80_q80_strict.first %t.q80_q80_strict.idempotent
// RUN: timeout 8 mlir-opt --canonicalize %t.q80_q80_strict.first > %t.q80_q80_strict.canonical
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_strict.first)" = "2 0 0"
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_strict.canonical)" = "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_strict.first)"

// q80_q80_partition: expand/collapse/transpose = 2/0/0.
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q80_q80_partition.ttshared.mlir > %t.q80_q80_partition.first
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q80_q80_partition.ttshared.mlir > %t.q80_q80_partition.second
// RUN: diff -u %t.q80_q80_partition.first %t.q80_q80_partition.second
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %t.q80_q80_partition.first > %t.q80_q80_partition.idempotent
// RUN: diff -u %t.q80_q80_partition.first %t.q80_q80_partition.idempotent
// RUN: timeout 8 mlir-opt --canonicalize %t.q80_q80_partition.first > %t.q80_q80_partition.canonical
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_partition.first)" = "2 0 0"
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_partition.canonical)" = "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_partition.first)"

// q40_q80_canonical: expand/collapse/transpose = 2/1/1.
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q40_q80_canonical.ttshared.mlir > %t.q40_q80_canonical.first
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q40_q80_canonical.ttshared.mlir > %t.q40_q80_canonical.second
// RUN: diff -u %t.q40_q80_canonical.first %t.q40_q80_canonical.second
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %t.q40_q80_canonical.first > %t.q40_q80_canonical.idempotent
// RUN: diff -u %t.q40_q80_canonical.first %t.q40_q80_canonical.idempotent
// RUN: timeout 8 mlir-opt --canonicalize %t.q40_q80_canonical.first > %t.q40_q80_canonical.canonical
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q40_q80_canonical.first)" = "2 1 1"
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q40_q80_canonical.canonical)" = "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q40_q80_canonical.first)"

// q4k_q8k_canonical: expand/collapse/transpose = 4/5/8.
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q4k_q8k_canonical.ttshared.mlir > %t.q4k_q8k_canonical.first
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q4k_q8k_canonical.ttshared.mlir > %t.q4k_q8k_canonical.second
// RUN: diff -u %t.q4k_q8k_canonical.first %t.q4k_q8k_canonical.second
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %t.q4k_q8k_canonical.first > %t.q4k_q8k_canonical.idempotent
// RUN: diff -u %t.q4k_q8k_canonical.first %t.q4k_q8k_canonical.idempotent
// RUN: timeout 8 mlir-opt --canonicalize %t.q4k_q8k_canonical.first > %t.q4k_q8k_canonical.canonical
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q4k_q8k_canonical.first)" = "4 5 8"
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q4k_q8k_canonical.canonical)" = "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q4k_q8k_canonical.first)"

// iq2xxs_q8k_canonical: expand/collapse/transpose = 5/7/5.
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/iq2xxs_q8k_canonical.ttshared.mlir > %t.iq2xxs_q8k_canonical.first
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/iq2xxs_q8k_canonical.ttshared.mlir > %t.iq2xxs_q8k_canonical.second
// RUN: diff -u %t.iq2xxs_q8k_canonical.first %t.iq2xxs_q8k_canonical.second
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %t.iq2xxs_q8k_canonical.first > %t.iq2xxs_q8k_canonical.idempotent
// RUN: diff -u %t.iq2xxs_q8k_canonical.first %t.iq2xxs_q8k_canonical.idempotent
// RUN: timeout 8 mlir-opt --canonicalize %t.iq2xxs_q8k_canonical.first > %t.iq2xxs_q8k_canonical.canonical
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.iq2xxs_q8k_canonical.first)" = "5 7 5"
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.iq2xxs_q8k_canonical.canonical)" = "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.iq2xxs_q8k_canonical.first)"

// synthetic_l3_canonical: expand/collapse/transpose = 6/0/0.
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/synthetic_l3_canonical.ttshared.mlir > %t.synthetic_l3_canonical.first
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/synthetic_l3_canonical.ttshared.mlir > %t.synthetic_l3_canonical.second
// RUN: diff -u %t.synthetic_l3_canonical.first %t.synthetic_l3_canonical.second
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %t.synthetic_l3_canonical.first > %t.synthetic_l3_canonical.idempotent
// RUN: diff -u %t.synthetic_l3_canonical.first %t.synthetic_l3_canonical.idempotent
// RUN: timeout 8 mlir-opt --canonicalize %t.synthetic_l3_canonical.first > %t.synthetic_l3_canonical.canonical
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.synthetic_l3_canonical.first)" = "6 0 0"
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.synthetic_l3_canonical.canonical)" = "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.synthetic_l3_canonical.first)"

// q80_q80_layout_trans_reshape: expand/collapse/transpose = 2/1/2.
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_q80_layout_trans_reshape.ttshared.mlir > %t.q80_q80_layout_trans_reshape.first
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_q80_layout_trans_reshape.ttshared.mlir > %t.q80_q80_layout_trans_reshape.second
// RUN: diff -u %t.q80_q80_layout_trans_reshape.first %t.q80_q80_layout_trans_reshape.second
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %t.q80_q80_layout_trans_reshape.first > %t.q80_q80_layout_trans_reshape.idempotent
// RUN: diff -u %t.q80_q80_layout_trans_reshape.first %t.q80_q80_layout_trans_reshape.idempotent
// RUN: timeout 8 mlir-opt --canonicalize %t.q80_q80_layout_trans_reshape.first > %t.q80_q80_layout_trans_reshape.canonical
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_layout_trans_reshape.first)" = "2 1 2"
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_layout_trans_reshape.canonical)" = "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_layout_trans_reshape.first)"

// q80_q80_layout_reshape_trans_reshape: expand/collapse/transpose = 2/1/2.
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_q80_layout_reshape_trans_reshape.ttshared.mlir > %t.q80_q80_layout_reshape_trans_reshape.first
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %S/Inputs/q80_q80_layout_reshape_trans_reshape.ttshared.mlir > %t.q80_q80_layout_reshape_trans_reshape.second
// RUN: diff -u %t.q80_q80_layout_reshape_trans_reshape.first %t.q80_q80_layout_reshape_trans_reshape.second
// RUN: timeout 60 buddy-opt --qgemm-simplify-dim-transforms %t.q80_q80_layout_reshape_trans_reshape.first > %t.q80_q80_layout_reshape_trans_reshape.idempotent
// RUN: diff -u %t.q80_q80_layout_reshape_trans_reshape.first %t.q80_q80_layout_reshape_trans_reshape.idempotent
// RUN: timeout 8 mlir-opt --canonicalize %t.q80_q80_layout_reshape_trans_reshape.first > %t.q80_q80_layout_reshape_trans_reshape.canonical
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_layout_reshape_trans_reshape.first)" = "2 1 2"
// RUN: test "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_layout_reshape_trans_reshape.canonical)" = "$(awk '/tensor\.expand_shape/ {e++} /tensor\.collapse_shape/ {c++} /linalg\.transpose/ {t++} END {print e + 0, c + 0, t + 0}' %t.q80_q80_layout_reshape_trans_reshape.first)"
