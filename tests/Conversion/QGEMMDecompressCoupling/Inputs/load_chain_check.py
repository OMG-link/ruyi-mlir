#!/usr/bin/env python3
# Licensed under the Apache License v2.0. See LICENSE for details.
#
# Compare the operand load chains of two pretty-printed IRs, or report the
# chains of one. Used by load_chain_pre.mlir (Req 12.21): under `pre` every
# load chain must be exactly what the recognizer produced, while the rest of
# the operator may well have been rebuilt, so a whole-IR topology comparison
# cannot express the requirement.
#
# A chain op is reduced to a signature that holds everything the IR states
# about it except SSA names: its kind, the kind of the op producing its first
# operand, its qgemm.node stage, the offsets/sizes/strides list of a view and
# the full type suffix of the line. Comparing the multiset of signatures
# therefore detects any change of shape, layout, stage, chain structure or op
# count, and ignores renumbering caused by changes elsewhere in the function.
#
# Usage: load_chain_check.py --same <reference.mlir>, output IR on stdin.
#
# Prints "LOAD CHAINS SAME <n>" / "LOAD CHAINS DIFFER" plus one VIOLATION line
# per differing signature, and exits non-zero on a difference.
#
# Usage: load_chain_check.py --forwarded <reference.mlir>, output IR on stdin.
#
# Checks Load_Copy_Forwarding under `pre` (Req 12.21, 14.9): the output has
# no annotated memref.alloc or memref.copy, every annotated
# bufferization.to_tensor reads a memref view op directly and is not
# writable, the view ops are exactly the reference's view ops, and the
# to_tensor result types are exactly the reference's, so no op of a chain
# gained a parallel dimension. The reference is the recognize-only output,
# which must contain at least one copied load for the check to mean
# anything. Prints "LOAD CHAINS FORWARDED <n>" / "LOAD CHAINS NOT FORWARDED"
# plus one VIOLATION line per failure, and exits non-zero on a failure.

import re
import sys

KINDS = ("bufferization.to_tensor", "memref.alloc", "memref.copy",
         "memref.reinterpret_cast", "memref.cast", "memref.subview",
         "memref.expand_shape", "memref.collapse_shape")


def chains(text):
    """Signature multiset of the load-chain ops of `text`, keyed by name so a
    producer can be named by kind rather than by SSA name."""
    producer = {}
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        kind = next((k for k in KINDS if (" = " + k) in stripped or
                     stripped.startswith(k)), None)
        if not kind:
            continue
        result = re.match(r"(%[\w]+) = ", stripped)
        if result:
            producer[result.group(1)] = kind
        lines.append((kind, stripped))

    signatures = []
    for kind, stripped in lines:
        body = stripped.split(" = ", 1)[1] if " = " in stripped else stripped
        operand = re.search(r"%[\w]+", body)
        stage = re.search(r'stage = "(\w+)"', stripped)
        # A dynamic entry is an SSA name, which renumbers when the rebuild
        # changes the op count elsewhere; keep only the fact that it is one.
        entries = [re.sub(r"%[\w]+", "%", e) for e in
                   re.findall(r"(?:offset|sizes|strides): \[[^\]]*\]", stripped)]
        flags = tuple(f for f in ("restrict", "writable") if
                      re.search(r"\b%s\b" % f, stripped))
        types = stripped.rsplit(" : ", 1)[1] if " : " in stripped else ""
        signatures.append((kind,
                           producer.get(operand.group(0), "outside")
                           if operand else "none",
                           stage.group(1) if stage else "none",
                           tuple(entries), flags, types))
    return signatures


VIEWS = KINDS[3:]
def forwarded(ref, got):
    violations = []
    annotated = [g for g in got if g[2] != "none"]
    for g in annotated:
        if g[0] in ("memref.alloc", "memref.copy"):
            violations.append("private copy kept: %s" % (g,))
        if g[0] == "bufferization.to_tensor":
            if g[1] not in VIEWS:
                violations.append("to_tensor does not read a view: %s" % (g,))
            if "writable" in g[4]:
                violations.append("to_tensor still writable: %s" % (g,))
    if not any(r[0] == "memref.copy" for r in ref):
        violations.append("reference has no copied load")
    ref_views = sorted((r for r in ref if r[0] in VIEWS), key=str)
    got_views = sorted((g for g in got if g[0] in VIEWS), key=str)
    if ref_views != got_views:
        violations.append("view ops differ: %s vs %s" % (ref_views, got_views))
    tensor_type = lambda sig: sig[5].rsplit(" to ", 1)[-1]
    ref_tt = sorted(tensor_type(r) for r in ref
                    if r[0] == "bufferization.to_tensor")
    got_tt = sorted(tensor_type(g) for g in got
                    if g[0] == "bufferization.to_tensor")
    if ref_tt != got_tt:
        violations.append("to_tensor types differ: %s vs %s" % (ref_tt, got_tt))
    for violation in violations:
        print("VIOLATION: " + violation)
    print("LOAD CHAINS %s %d" % ("NOT FORWARDED" if violations else
                                 "FORWARDED", len(annotated)))
    return 1 if violations else 0
def main():
    mode = sys.argv[1]
    got = chains(sys.stdin.read())
    if mode == "--forwarded":
        return forwarded(chains(open(sys.argv[2]).read()), got)
    if mode != "--same":
        sys.stderr.write("unknown mode %s\n" % mode)
        return 2
    ref = chains(open(sys.argv[2]).read())
    violations = []
    for signature in sorted(set(ref) | set(got), key=str):
        a, b = ref.count(signature), got.count(signature)
        if a != b:
            violations.append("%d reference vs %d output: %s" % (a, b, signature))
    for violation in violations:
        print("VIOLATION: " + violation)
    print("LOAD CHAINS %s %d" % ("DIFFER" if violations else "SAME", len(got)))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
