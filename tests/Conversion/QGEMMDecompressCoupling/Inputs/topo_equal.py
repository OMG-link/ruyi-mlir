#!/usr/bin/env python3
# ===- topo_equal.py - Topology checker for QGEMM lit tests ----------------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===-----------------------------------------------------------------------===//
#
# Decides whether two IRs are Topologically_Identical: their op graphs are
# isomorphic modulo the order of ops within a block and SSA value names.
#
# Usage: topo_equal.py [--expect-different] A.mlir B.mlir
#
# Both files must be printed with `--mlir-print-op-generic` (ideally also
# `--mlir-print-local-scope`). Each op is hashed bottom-up from its name,
# properties, attribute dictionary, signature types, the hashes of its
# operands (result index included) and its regions. Block arguments hash by
# their owner block's position (nesting depth, block index) and arg index.
# Each block is compared as a multiset of op hashes. Exit status 0 means
# equal; 1 means different, and the first unmatched op of each side is
# printed; 2 means a parse error or any other error. With
# --expect-different the verdict is inverted for negative assertions: 0 only
# when both inputs parsed and differ, 1 when they are identical, 2 on error.
#
# ===-----------------------------------------------------------------------===//

import hashlib
import re
import sys
from collections import Counter


class ParseError(Exception):
    pass


class Op:
    def __init__(self):
        self.results = []  # SSA names, one per result
        self.name = ""
        self.operands = []  # SSA refs, e.g. "%3" or "%3#1"
        self.successors = ""
        self.props = ""
        self.attrs = ""
        self.sig = ""
        self.regions = []  # list[list[Block]]
        self.text = ""  # first source line, for diagnostics
        self.hash = None


class Block:
    def __init__(self):
        self.args = []  # list[(name, type)]
        self.ops = []
        self.hash = None


class Parser:
    def __init__(self, text):
        self.s = text
        self.i = 0

    def ws(self):
        while self.i < len(self.s):
            c = self.s[self.i]
            if c.isspace():
                self.i += 1
            elif self.s.startswith("//", self.i):
                j = self.s.find("\n", self.i)
                self.i = len(self.s) if j < 0 else j
            else:
                break

    def peek(self, t):
        self.ws()
        return self.s.startswith(t, self.i)

    def expect(self, t):
        if not self.peek(t):
            ctx = self.s[self.i : self.i + 60]
            raise ParseError("expected %r at offset %d: %r" % (t, self.i, ctx))
        self.i += len(t)

    def balanced(self, open_c, close_c):
        """Return the text of a balanced bracket group starting at the cursor,
        skipping over string literals."""
        self.ws()
        assert self.s[self.i] == open_c
        depth = 0
        start = self.i
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == '"':
                self.string()
                continue
            if c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    self.i += 1
                    return self.s[start : self.i]
            self.i += 1
        raise ParseError("unbalanced %s" % open_c)

    def string(self):
        assert self.s[self.i] == '"'
        start = self.i
        self.i += 1
        while self.s[self.i] != '"':
            if self.s[self.i] == "\\":
                self.i += 1
            self.i += 1
        self.i += 1
        return self.s[start : self.i]

    def ssa_name(self):
        self.ws()
        m = re.compile(r"%[\w$.\-]+(:\d+|#\d+)?").match(self.s, self.i)
        if not m:
            raise ParseError("expected SSA value at offset %d" % self.i)
        self.i = m.end()
        return m.group(0)

    def rest_of_line(self):
        j = self.s.find("\n", self.i)
        j = len(self.s) if j < 0 else j
        t = self.s[self.i : j].strip()
        self.i = j
        return t

    def parse_op(self):
        self.ws()
        op = Op()
        line_end = self.s.find("\n", self.i)
        op.text = self.s[self.i : line_end if line_end >= 0 else None].strip()
        if self.peek("%"):
            while True:
                r = self.ssa_name()
                if ":" in r:
                    base, n = r.split(":")
                    op.results += ["%s#%d" % (base, k) for k in range(int(n))]
                    op.results.append(base)  # bare name aliases #0
                else:
                    op.results.append(r)
                if self.peek(","):
                    self.expect(",")
                    continue
                break
            self.expect("=")
        self.ws()
        if self.s[self.i] != '"':
            raise ParseError("expected generic op name at offset %d: %r"
                             % (self.i, self.s[self.i : self.i + 60]))
        op.name = self.string()[1:-1]
        self.expect("(")
        while not self.peek(")"):
            op.operands.append(self.ssa_name())
            if self.peek(","):
                self.expect(",")
        self.expect(")")
        if self.peek("["):
            op.successors = self.balanced("[", "]")
        if self.peek("<{"):
            self.ws()
            self.i += 1  # skip '<'
            op.props = self.balanced("{", "}")
            self.expect(">")
        if self.peek("("):
            self.expect("(")
            while True:
                op.regions.append(self.parse_region())
                if self.peek(","):
                    self.expect(",")
                    continue
                break
            self.expect(")")
        if self.peek("{"):
            op.attrs = self.balanced("{", "}")
        self.expect(":")
        op.sig = self.rest_of_line()
        return op

    def parse_region(self):
        self.expect("{")
        blocks = []
        cur = None
        while not self.peek("}"):
            if self.peek("^"):
                cur = Block()
                blocks.append(cur)
                m = re.compile(r"\^[\w$.\-]+").match(self.s, self.i)
                self.i = m.end()
                if self.peek("("):
                    self.expect("(")
                    while not self.peek(")"):
                        n = self.ssa_name()
                        self.expect(":")
                        self.ws()
                        # A block arg type ends at the next top-level ',' or ')'.
                        start, depth = self.i, 0
                        while True:
                            c = self.s[self.i]
                            if c in "<([{":
                                depth += 1
                            elif c in ">)]}":
                                if depth == 0:
                                    break
                                depth -= 1
                            elif c == "," and depth == 0:
                                break
                            self.i += 1
                        t = self.s[start : self.i].strip()
                        t = re.sub(r"\s+loc\(.*\)$", "", t)
                        cur.args.append((n, t))
                        if self.peek(","):
                            self.expect(",")
                    self.expect(")")
                self.expect(":")
                continue
            if cur is None:
                cur = Block()
                blocks.append(cur)
            cur.ops.append(self.parse_op())
        self.expect("}")
        return blocks


def H(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:24]


class Hasher:
    def __init__(self, top_ops):
        self.defs = {}  # ssa name -> ("op", Op, idx) | ("arg", key)
        self.top = top_ops
        self._index(top_ops, 0)

    def _index(self, ops, depth):
        for op in ops:
            for idx, r in enumerate(op.results):
                if "#" in r:
                    self.defs[r] = ("op", op, int(r.split("#")[1]))
                elif r not in self.defs:
                    self.defs[r] = ("op", op, 0)
            for ri, region in enumerate(op.regions):
                for bi, blk in enumerate(region):
                    for ai, (n, t) in enumerate(blk.args):
                        self.defs[n] = ("arg", (depth + 1, ri, bi, ai, t))
                    self._index(blk.ops, depth + 1)

    def value(self, ref):
        d = self.defs.get(ref)
        if d is None:
            raise ParseError("use of undefined value %s" % ref)
        if d[0] == "arg":
            return H("arg", d[1])
        return H("res", self.op(d[1]), d[2])

    def op(self, op):
        if op.hash is not None:
            return op.hash
        op.hash = "pending"
        ops = [self.value(v) for v in op.operands]
        regs = [[self.block(b) for b in r] for r in op.regions]
        op.hash = H(op.name, op.props, op.attrs, op.sig, op.successors,
                    len(op.results), ops, regs)
        return op.hash

    def block(self, b):
        if b.hash is None:
            b.hash = H([t for _, t in b.args],
                       sorted(Counter(self.op(o) for o in b.ops).items()))
        return b.hash


def load(path):
    with open(path) as f:
        text = f.read()
    p = Parser(text)
    ops = []
    while True:
        p.ws()
        if p.i >= len(p.s):
            break
        ops.append(p.parse_op())
    hasher = Hasher(ops)
    for o in ops:
        hasher.op(o)
    return ops


def first_mismatch(ops_a, ops_b):
    """Return (opA, opB) describing the first unmatched op of each side within
    one op list, descending into same-kind region-carrying op pairs."""
    ca = Counter(o.hash for o in ops_a)
    cb = Counter(o.hash for o in ops_b)
    only_a = [o for o in ops_a if ca[o.hash] > cb.get(o.hash, 0)]
    only_b = [o for o in ops_b if cb[o.hash] > ca.get(o.hash, 0)]
    for oa in only_a:
        if not oa.regions:
            continue
        # Descend only into an unambiguous counterpart; otherwise report here.
        cands = [ob for ob in only_b
                 if ob.name == oa.name and ob.props == oa.props
                 and ob.attrs == oa.attrs and ob.sig == oa.sig
                 and len(ob.regions) == len(oa.regions)]
        if oa.name in ("builtin.module", "func.func") and len(cands) == 1:
            for ra, rb in zip(oa.regions, cands[0].regions):
                for ba, bb in zip(ra, rb):
                    if ba.hash != bb.hash:
                        return first_mismatch(ba.ops, bb.ops)
    # Prefer an unmatched op none of whose operand producers in this block is
    # itself unmatched: that op is where the two graphs start to diverge.
    def pick(only, ops):
        bad = set(id(o) for o in only)
        names = {}
        for o in ops:
            for r in o.results:
                names[r] = o
        for o in only:
            if not any(id(names[v]) in bad for v in o.operands if v in names):
                return o
        return only[0] if only else None
    return pick(only_a, ops_a), pick(only_b, ops_b)


def compare(argv):
    """Return 0 when identical, 1 when different, 2 on any error."""
    if len(argv) != 3:
        print("usage: topo_equal.py [--expect-different] A.mlir B.mlir",
              file=sys.stderr)
        return 2
    try:
        a, b = load(argv[1]), load(argv[2])
    except ParseError as e:
        print("topo_equal: parse error: %s" % e, file=sys.stderr)
        return 2
    if not a or not b:
        print("topo_equal: empty input", file=sys.stderr)
        return 2
    if sorted(o.hash for o in a) == sorted(o.hash for o in b):
        return 0
    oa, ob = first_mismatch(a, b)
    print("topo_equal: IRs are not topologically identical", file=sys.stderr)
    print("  first unmatched op in %s:\n    %s"
          % (argv[1], oa.text if oa else "<none>"), file=sys.stderr)
    print("  first unmatched op in %s:\n    %s"
          % (argv[2], ob.text if ob else "<none>"), file=sys.stderr)
    return 1


def main(argv):
    # --expect-different inverts the verdict: exit 0 only when both inputs
    # parsed and differ, 1 when they are identical, 2 on any error. A parse
    # error or crash therefore never passes as "different".
    expect_different = "--expect-different" in argv[1:]
    args = [argv[0]] + [a for a in argv[1:] if a != "--expect-different"]
    try:
        rc = compare(args)
    except Exception as e:  # crashes are errors, never a verdict
        print("topo_equal: internal error: %r" % (e,), file=sys.stderr)
        return 2
    if not expect_different:
        return rc
    if rc == 1:
        print("topo_equal: IRs differ, as expected", file=sys.stderr)
        return 0
    if rc == 0:
        print("topo_equal: IRs are identical but were expected to differ",
              file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
