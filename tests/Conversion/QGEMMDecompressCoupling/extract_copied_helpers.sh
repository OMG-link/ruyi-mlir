#!/usr/bin/env bash
# Prints the DimMapping alias and the bodies of the helpers that
# QGEMMDecompressCoupling/AxisTracker.cpp copies from
# QGEMMRecognize/Recognition.cpp, in a fixed order.
#
# Usage: extract_copied_helpers.sh <cpp-file> [<header-with-DimMapping>]
#        extract_copied_helpers.sh --list-region <cpp-file>
#        extract_copied_helpers.sh --list-expected
# If the file contains "Begin copy"/"End copy" markers, only that region is
# searched for the functions. DimMapping is taken from the header if given.
# --list-region prints the sorted names of every static function defined in
# the marked region (it fails if the file has no markers); --list-expected
# prints the sorted list of functions compared here. Diffing the two catches a
# helper added to the copy without being compared.
set -euo pipefail
FNS="getRank makeIdentityMapping makeNullMapping resizeMapping
mapThroughAffineMap getIterationMapping getProducerMapping"
if [ "${1:-}" = "--list-expected" ]; then
  printf '%s\n' $FNS | sort
  exit 0
fi
if [ "${1:-}" = "--list-region" ]; then
  src="$2"
  grep -q 'Begin copy of QGEMMRecognize/Recognition.cpp' "$src" || {
    echo "error: no Begin copy marker in $src" >&2
    exit 1
  }
  # A static function definition starts with `static` at column 0. Its name
  # is the identifier before the first `(` on that line or, when the return
  # type stands on its own line, at the start of the next line.
  awk '
    /Begin copy of QGEMMRecognize\/Recognition.cpp/ { inside = 1; next }
    /End copy of QGEMMRecognize\/Recognition.cpp/   { inside = 0; next }
    !inside { next }
    pending {
      if (match($0, /^[A-Za-z_][A-Za-z0-9_]*\(/))
        print substr($0, 1, RLENGTH - 1)
      else
        print "<unparsed static definition>"
      pending = 0
      next
    }
    /^static / {
      if (match($0, /[A-Za-z_][A-Za-z0-9_]*\(/))
        print substr($0, RSTART, RLENGTH - 1)
      else
        pending = 1
    }
  ' "$src" | sort
  exit 0
fi
src="$1"
alias_src="${2:-$1}"
grep -E '^using DimMapping = ' "$alias_src"
region=$(awk '
  /Begin copy of QGEMMRecognize\/Recognition.cpp/ { marked = 1; inside = 1; next }
  /End copy of QGEMMRecognize\/Recognition.cpp/   { inside = 0; next }
  { lines[NR] = $0; flag[NR] = inside }
  END { for (i = 1; i <= NR; i++) if (!marked || flag[i]) print lines[i] }
' "$src")
for fn in $FNS; do
  out=$(printf '%s\n' "$region" | awk -v fn="$fn" '
    !on && $0 ~ ("^static .*[ *&]" fn "\\(") { on = 1 }
    on { print }
    on && /^}/ { exit }
  ')
  if [ -z "$out" ]; then
    echo "error: function $fn not found in $src" >&2
    exit 1
  fi
  printf '%s\n' "$out"
done
