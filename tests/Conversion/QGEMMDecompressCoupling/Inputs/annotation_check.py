# Checks annotation completeness (Property 10) on the pretty-printed output of
# --qgemm-decompress-coupling read from stdin. argv[1] is the output of the
# same pipeline without the coupling pass (the reference annotation).
#
# Verified:
#  1. Every qgemm.node carries a root id that the reference also has, and a
#     stage in {data_prep, compute}.
#  2. Stage rule: an annotated op is `compute` iff its result carries both the
#     m and the n axis, or it is a Pure_Broadcast; `data_prep` otherwise.
#     m/n axes are identified structurally: seeded at m_dim / n_dim of the
#     qgemm.root op and propagated backwards to operands through the
#     linalg.generic indexing maps, linalg.reduce dimensions, linalg.transpose
#     permutations, expand/collapse_shape reassociations (a merged group's
#     label goes to its outermost non-unit axis), tensor.reshape (equal-size
#     prefix/suffix axes) and extract/insert_slice (size-aligned, unit dims
#     dropped). Matching by extent, as check_pre_invariant.py does, misfires on
#     q80 where the block size 32 equals n. An op whose axes are not reached
#     this way counts as carrying neither.
#     A Pure_Broadcast is a single-input linalg.generic with an identity output
#     map whose body is a bare `linalg.yield %in` and whose omitted dim is m
#     or n; block-structure broadcasts follow the plain axis rule.
#  3. No tensor.empty, linalg.fill or constant carries qgemm.node.
#  4. Every reference root has exactly one qgemm.root in the output, with the
#     reference m_dim / n_dim.
# Prints "ANNOTATION OK <n-annotated> <n-pb>" or one "VIOLATION: ..." line per
# failure, and exits non-zero on a violation.
import re
import sys

node_re = re.compile(r'qgemm\.node = \{root = (\d+) : i64, stage = "(\w+)"\}')
root_re = re.compile(r"qgemm\.root = \{m_dim = (\d+) : i64, n_dim = (\d+) : i64\}")


def result_shape(line):
    # Generic / named linalg ops: the outs type is the result type.
    m = re.search(r"outs\(%[\w]+ : tensor<([0-9x]*)x\w+>\)", line)
    if m is None:
        types = re.findall(r"tensor<([0-9x]*)x\w+>", line)
        if not types:
            return None
        m_last = types[-1]
        return [int(x) for x in m_last.split("x") if x]
    return [int(x) for x in m.group(1).split("x") if x]


def roots(text):
    out = {}
    for l in text.splitlines():
        r = root_re.search(l)
        if r:
            n = node_re.search(l)
            out.setdefault(n.group(1), []).append(
                (int(r.group(1)), int(r.group(2)), result_shape(l)))
    return out



def analyze(text):
    """Parse the annotated ops of pretty-printed `text` and label their m/n
    axes. Returns (lines, ops, labels), labels mapping every labelled value
    name to {axis: "m"|"n"}. Shared with compute_core_check.py."""
    lines = text.splitlines()
    def amap(s):
        s = s.strip()
        m = re.search(r"^%s = affine_map<\(([^)]*)\) -> \(([^)]*)\)>" % re.escape(s),
                      text, re.M) if s.startswith("#") else \
            re.match(r"affine_map<\(([^)]*)\) -> \(([^)]*)\)>", s)
        split = lambda x: [d.strip() for d in x.split(",") if d.strip()]
        return split(m.group(1)), split(m.group(2))


    def align(full, part):
        # Map axes of `full` to axes of the rank-reduced (or equal-rank) `part`
        # by greedily matching sizes; unmatched full axes are dropped unit dims.
        out, j = {}, 0
        for a, e in enumerate(full):
            if j < len(part) and (len(part) == len(full) or part[j] == e):
                out[a] = j
                j += 1
        return out


    # Collect annotated ops: result name, kind, operands with their maps.
    ops = []
    for i, l in enumerate(lines):
        n = node_re.search(l)
        if not n:
            continue
        res = re.match(r"^\s*(%[\w]+)(?::\d+)? = ", l)
        op = dict(line=i, root=n.group(1), stage=n.group(2),
                  result=res.group(1) if res else None, kind=None)
        g = re.match(r"^\s*%[\w]+ = linalg\.generic \{indexing_maps = \[(.*?)\]"
                     r".*?\}(?: ins\((.*?) : .*?\))? outs\((%[\w]+) : ", l)
        r = re.match(r"^\s*%[\w]+ = linalg\.reduce ins\((%[\w]+) : .*?\) "
                     r"outs\((%[\w]+) : .*?dimensions = \[([0-9, ]*)\]", l)
        if g:
            op["kind"] = "generic"
            op["maps"] = [amap(x) for x in re.split(r",\s*(?=#|affine_map)", g.group(1))]
            op["ins"] = [x.strip() for x in (g.group(2) or "").split(",") if x.strip()]
            op["outs"] = g.group(3)
            body = []
            j = i + 1
            while not lines[j].lstrip().startswith("} ->"):
                if not lines[j].lstrip().startswith("^bb"):
                    body.append(lines[j].strip())
                j += 1
            op["body"] = body
        elif r:
            op["kind"] = "reduce"
            op["ins"] = [r.group(1)]
            op["outs"] = r.group(2)
            op["red"] = [int(x) for x in r.group(3).split(",") if x.strip()]
        else:
            t = re.match(r"^\s*%[\w]+ = linalg\.transpose ins\((%[\w]+) : .*?"
                         r"permutation = \[([0-9, ]*)\]", l)
            c = re.match(r"^\s*%[\w]+ = tensor\.(collapse|expand)_shape (%[\w]+) "
                         r"(\[\[.*?\]\])", l)
            x = re.match(r"^\s*%[\w]+ = tensor\.extract_slice (%[\w]+)\[", l)
            y = re.match(r"^\s*%[\w]+ = tensor\.insert_slice (%[\w]+) into (%[\w]+)\[", l)
            rs = re.match(r"^\s*%[\w]+ = tensor\.reshape (%[\w]+)\(", l)
            # An operand load chain's bufferization.to_tensor keeps the rank
            # and dimension order of its memref operand, so the axes carry
            # straight through to the memref view op that produced it.
            # Without this the view op would look axis-free.
            tt = re.match(r"^\s*%[\w]+ = bufferization\.to_tensor (%[\w]+)", l)
            shapes = [[int(v) for v in sh.split("x") if v]
                      for sh in re.findall(r"tensor<([0-9x]*)x\w+>", l)]
            if t:
                op["kind"] = "transpose"
                op["ins"] = [t.group(1)]
                op["perm"] = [int(v) for v in t.group(2).split(",")]
            elif c:
                op["kind"] = c.group(1)
                op["ins"] = [c.group(2)]
                op["reassoc"] = [[int(v) for v in g.split(",") if v.strip()]
                                 for g in re.findall(r"\[([0-9, ]*)\]", c.group(3)[1:-1])]
                op["src_shape"] = shapes[0]
            elif rs:
                # tensor.reshape: keep the equal-size prefix and suffix axes.
                op["kind"] = "slice"
                op["ins"] = [rs.group(1)]
                src, dst = shapes[0], shapes[-1]
                al = {}
                for a in range(min(len(src), len(dst))):
                    if src[a] != dst[a]:
                        break
                    al[a] = a
                for a in range(1, min(len(src), len(dst)) + 1):
                    if src[-a] != dst[-a]:
                        break
                    al[len(dst) - a] = len(src) - a
                op["align"] = [al]
            elif tt and shapes:
                op["kind"] = "slice"
                op["ins"] = [tt.group(1)]
                op["align"] = [{a: a for a in range(len(shapes[-1]))}]
            elif x and len(shapes) == 2:
                # extract_slice: source shapes[0] -> result shapes[1].
                op["kind"] = "slice"
                op["ins"] = [x.group(1)]
                op["align"] = [align(shapes[1], shapes[0])]
            elif y and len(shapes) == 2:
                # insert_slice: source shapes[0] into dest/result shapes[1].
                op["kind"] = "slice"
                op["ins"] = [y.group(1), y.group(2)]
                op["align"] = [align(shapes[1], shapes[0]),
                               {a: a for a in range(len(shapes[1]))}]
        ops.append(op)

    # Axis labels: result name -> {axis: "m"|"n"}. Seeded at each root's result
    # and propagated backwards to operands through the indexing maps, so m/n are
    # identified structurally rather than by extent.
    labels = {}
    for op in reversed(ops):
        lab = dict(labels.get(op["result"], {}))
        r = root_re.search(lines[op["line"]])
        if r:
            lab[int(r.group(1))] = "m"
            lab[int(r.group(2))] = "n"
        op["labels"] = lab
        def give(name, axes):
            if axes:
                labels.setdefault(name, {}).update(axes)
        if op["kind"] == "generic":
            dims, out = op["maps"][-1]
            dl = {out[p]: v for p, v in lab.items() if p < len(out)}
            for name, (_, res) in zip(op["ins"] + [op["outs"]], op["maps"]):
                give(name, {j: dl[d] for j, d in enumerate(res) if d in dl})
        elif op["kind"] == "reduce":
            keep = [a for a in range(len(lab) + 64) if a not in op["red"]]
            give(op["ins"][0], {keep[p]: v for p, v in lab.items()})
            give(op["outs"], lab)
        elif op["kind"] == "transpose":
            give(op["ins"][0], {op["perm"][p]: v for p, v in lab.items()})
        elif op["kind"] == "expand":
            give(op["ins"][0], {gi: lab[a] for gi, g in enumerate(op["reassoc"])
                                for a in g if a in lab})
        elif op["kind"] == "collapse":
            axes = {}
            for p, v in lab.items():
                g = op["reassoc"][p]
                big = [a for a in g if op["src_shape"][a] != 1]
                # A labeled group goes to its outermost non-unit source axis
                # (row-major: the m/n axis is the major one of a merged group).
                axes[(big or g)[0]] = v
            give(op["ins"][0], axes)
        elif op["kind"] == "slice":
            for name, al in zip(op["ins"], op["align"]):
                give(name, {al[p]: v for p, v in lab.items() if p in al})

    return lines, ops, labels


def main():
    ref = roots(open(sys.argv[1]).read())
    text = sys.stdin.read()
    lines = text.splitlines()
    got = roots(text)
    violations = []

    for rid, entries in ref.items():
        g = got.get(rid, [])
        if len(g) != 1:
            violations.append("root %s: %d qgemm.root attributes" % (rid, len(g)))
        elif g[0][:2] != entries[0][:2]:
            violations.append("root %s: qgemm.root %s, expected %s" %
                              (rid, g[0][:2], entries[0][:2]))
    for rid in got:
        if rid not in ref:
            violations.append("root %s: qgemm.root not in reference" % rid)

    lines, ops, _ = analyze(text)
    num, num_pb = 0, 0
    for op in ops:
        num += 1
        l = lines[op["line"]]
        rid, stage = op["root"], op["stage"]
        where = "line %d" % (op["line"] + 1)
        if re.search(r"= (tensor\.empty|linalg\.fill|arith\.constant)\b", l):
            violations.append("%s: init op carries qgemm.node" % where)
            continue
        if stage not in ("data_prep", "compute"):
            violations.append("%s: stage %s" % (where, stage))
            continue
        if rid not in ref:
            violations.append("%s: unknown root %s" % (where, rid))
            continue
        is_pb = False
        if op["kind"] == "generic" and len(op["ins"]) == 1:
            dims, out = op["maps"][-1]
            kept = op["maps"][0][1]
            is_pb = (out == dims and len(op["body"]) == 1 and
                     re.fullmatch(r"linalg\.yield %in : \S+", op["body"][0]) is not None)
            # Only m/n broadcasts are Pure_Broadcasts in the sense of Req 8.3;
            # block-structure broadcasts follow the plain axis rule.
            is_pb = is_pb and any(op["labels"].get(p) for p, d in enumerate(dims)
                                  if d not in kept)
        num_pb += is_pb
        carried = set(op["labels"].values())
        want = "compute" if carried == {"m", "n"} or is_pb else "data_prep"
        if stage != want:
            violations.append("%s: %s stage %s, expected %s (axes %s, pb %s)" %
                              (where, op["result"], stage, want, op["labels"], is_pb))

    for v in violations:
        print("VIOLATION: " + v)
    print("ANNOTATION %s %d %d" % ("FAIL" if violations else "OK", num, num_pb))
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
