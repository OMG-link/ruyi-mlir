# Checks the `decompress=pre` invariant (Property 5) on the pretty-printed
# output of --qgemm-decompress-coupling read from stdin.
#
# Verified, for every linalg.generic carrying qgemm.node:
#  1. Only Pure_Broadcasts introduce m or n. A generic "introduces" dim d when
#     d is a parallel iterator that appears in its output map but is omitted by
#     some data-operand (ins) map, and d is m or n. m/n are identified by
#     extent: an output axis whose size equals the m or n extent of the op's
#     root (read from the qgemm.root op's result type). This is a heuristic; a
#     k-side axis of the same size would be reported and must be inspected.
#     Introductions of other dims (block-structure broadcasts, codebook
#     expansion) are legitimate and ignored. A Pure_Broadcast
#     is a generic with exactly one input whose body is a bare
#     `linalg.yield %in`, whose output map is the identity and whose input map
#     omits exactly one parallel dimension (the dims it keeps stay in order).
#  2. Every Pure_Broadcast result is used, and every chain of Pure_Broadcasts
#     ends at a non-broadcast op: either an op of the same root that reads the
#     broadcast value (as input or as init) with every output parallel dim
#     present in its operand map
#     (it carries d, i.e. it is in V_d), or an Outside_User (an op of another
#     root, an op without qgemm.node, or the function terminator).
# Not verified: introductions by non-generic ops (linalg.broadcast etc.), which
# under the canonical captures do not occur after Separation.
# Prints "PRE-INVARIANT OK <n-generics> <n-broadcasts>" or one
# "VIOLATION: ..." line per failure, and exits non-zero on a violation.
import re
import sys

text = sys.stdin.read()
lines = text.splitlines()

aliases = {}
for l in lines:
    m = re.match(r"^(#map\d*) = affine_map<\(([^)]*)\) -> \(([^)]*)\)>", l)
    if m:
        aliases[m.group(1)] = (m.group(2), m.group(3))


def parse_map(s):
    s = s.strip()
    if s in aliases:
        dims, res = aliases[s]
    else:
        m = re.match(r"affine_map<\(([^)]*)\) -> \(([^)]*)\)>", s)
        dims, res = m.group(1), m.group(2)
    dims = [d.strip() for d in dims.split(",") if d.strip()]
    res = [r.strip() for r in res.split(",") if r.strip()]
    return dims, res


ops = []  # dicts: line, result, ins, maps, iters, root, is_pb, users
gen_re = re.compile(
    r"^\s*(%[\w]+)(?::\d+)? = linalg\.generic \{indexing_maps = \[(.*?)\], "
    r"iterator_types = \[(.*?)\]\} ins\((.*?) : .*?\) outs\((.*?) : "
    r"tensor<([0-9x]*)x\w+>\)")
i = 0
while i < len(lines):
    l = lines[i]
    m = gen_re.match(l)
    if m and "qgemm.node" in l:
        maps = [parse_map(s) for s in re.split(r",\s*(?=#|affine_map)", m.group(2))]
        iters = [t.strip().strip('"') for t in m.group(3).split(",")]
        ins = [x.strip() for x in m.group(4).split(",")]
        root = re.search(r"qgemm\.node = \{root = (\d+)", l).group(1)
        body = []
        j = i + 1
        while not lines[j].lstrip().startswith("} ->"):
            body.append(lines[j].strip())
            j += 1
        shape = [int(x) for x in m.group(6).split("x") if x]
        ops.append(dict(line=i + 1, result=m.group(1), ins=ins, maps=maps,
                        iters=iters, root=root, body=body, shape=shape,
                        outs=m.group(5).strip()))
        i = j
    i += 1

# Extents of m and n per root, read from the qgemm.root op's result type.
mn_ext = {}
for l in lines:
    r = re.search(r"qgemm\.node = \{root = (\d+).*qgemm\.root = "
                  r"\{m_dim = (\d+) : i64, n_dim = (\d+)", l)
    if r:
        outs = re.search(r"outs\(%[\w]+ : tensor<([0-9x]*)x\w+>\)", l)
        shape = [int(x) for x in outs.group(1).split("x")]
        mn_ext[r.group(1)] = {shape[int(r.group(2))], shape[int(r.group(3))]}

violations = []
by_result = {}
for op in ops:
    dims, out = op["maps"][-1]
    par_out = [r for r in out if r in dims and op["iters"][dims.index(r)] == "parallel"]
    intro = []
    for k, (_, res) in enumerate(op["maps"][:len(op["ins"])]):
        missing = [d for d in par_out if d not in res and
                   op["shape"][out.index(d)] in mn_ext[op["root"]]]
        if missing:
            intro.append((k, missing))
    body = [b for b in op["body"] if not b.startswith("^bb")]
    all_missing = [d for d in par_out if op["ins"] and d not in op["maps"][0][1]]
    op["is_pb"] = (len(op["ins"]) == 1 and out == dims and len(intro) == 1
                   and len(all_missing) == 1 and len(body) == 1
                   and re.fullmatch(r"linalg\.yield %in : \S+", body[0]) is not None
                   and [d for d in dims if d != intro[0][1][0]] == op["maps"][0][1])
    op["par_out"] = par_out
    if intro and not op["is_pb"]:
        violations.append("line %d: non-broadcast generic %s introduces %s" %
                          (op["line"], op["result"], intro))
    by_result[op["result"]] = op

# Req 9.4, second condition, on the memref side. A load chain's view op with a
# stride-0 dimension carries a parallel dimension that its operand does not,
# and it is not a Pure_Broadcast; only on-the-fly folds a parallel dimension
# into a view that way (Req 7.9), so under `pre` no such op exists.
for i, l in enumerate(lines):
    if "memref.reinterpret_cast" not in l or "qgemm.node" not in l:
        continue
    strides = re.search(r"strided<\[([-0-9, ?]*)\]", l)
    if not strides or not any(s.strip() == "0" for s in strides.group(1).split(",")):
        continue
    name = re.match(r"^\s*(%[\w]+) = ", l)
    violations.append("line %d: view %s introduces a parallel dimension at "
                      "stride 0" % (i + 1, name.group(1) if name else "?"))


def users_of(name):
    pat = re.compile(re.escape(name) + r"(?![\w#])")
    found = []
    for n, l in enumerate(lines):
        if "=" in l and pat.search(l.split("=", 1)[1]) or \
                ("=" not in l and pat.search(l)):
            found.append(n)
    return found


line_to_op = {op["line"] - 1: op for op in ops}
num_pb = 0
for op in ops:
    if not op["is_pb"]:
        continue
    num_pb += 1
    uses = users_of(op["result"])
    if not uses:
        violations.append("line %d: broadcast %s is unused" % (op["line"], op["result"]))
    for u in uses:
        ul = lines[u]
        uop = line_to_op.get(u)
        if uop is not None and uop["is_pb"]:
            continue  # chain continues
        if "qgemm.node" not in ul or uop is None and "return" in ul:
            continue  # Outside_User: no annotation / terminator
        if re.search(r"qgemm\.node = \{root = (\d+)", ul).group(1) != op["root"]:
            continue  # Outside_User: another root
        if uop is None:
            continue  # same-root non-generic consumer; carries d by its type
        operands = uop["ins"] + [uop["outs"]]  # an init counts like an input
        k = operands.index(op["result"])
        res = uop["maps"][k][1]
        if any(d not in res for d in uop["par_out"]):
            violations.append("line %d: broadcast %s ends at %s outside d space" %
                              (op["line"], op["result"], uop["result"]))

# 3. No decompression computation between an m/n broadcast and the multiply
#    (Req 12.14). From every m/n Pure_Broadcast, the value is followed through
#    same-root Pure_Broadcasts and pure dim transforms to the first other op E
#    that reads it as data (an `ins` operand; reading it as an init is not
#    computation on it). Under `pre` that op is a join: the multiply, or the
#    op where a broadcast factor meets a value in V_d. E must therefore have
#    another data operand that does not itself come, through pure dim
#    transforms, from a broadcast inserting the same loop dim of E. A
#    decompression op (extsi, andi, shift, lookup, a block-sum reduction ...)
#    reached by a broadcast fails this, which is what `on-the-fly` produces.
DIM_TRANSFORM_RE = re.compile(
    r"= (tensor\.(expand_shape|collapse_shape|extract_slice|insert_slice|"
    r"reshape)|linalg\.transpose)\b")
OPERAND_RE = re.compile(r"%[\w]+(?:#\d+)?")
defs = {}  # result name -> line index, for every annotated op
for n, l in enumerate(lines):
    m = re.match(r"^\s*(%[\w]+)(?::\d+)? = ", l)
    if m and "qgemm.node" in l:
        defs[m.group(1)] = n


def root_of(l):
    r = re.search(r"qgemm\.node = \{root = (\d+)", l)
    return r.group(1) if r else None


def inserted_loop_dim(pb, user):
    """Loop dim of generic `user` that broadcast `pb` (feeding it directly)
    inserts, or None when it cannot be told."""
    dims, out = pb["maps"][-1]
    kept = pb["maps"][0][1]
    ins_axes = [p for p, d in enumerate(out) if d not in kept]
    if len(ins_axes) != 1 or pb["result"] not in user["ins"]:
        return None
    res = user["maps"][user["ins"].index(pb["result"])][1]
    return res[ins_axes[0]] if ins_axes[0] < len(res) else None


def source_pb(name, seen=None):
    """The Pure_Broadcast `name` comes from through pure dim transforms."""
    seen = seen or set()
    while name in defs and name not in seen:
        seen.add(name)
        op = by_result.get(name)
        if op is not None:
            return op if op["is_pb"] else None
        l = lines[defs[name]]
        if not DIM_TRANSFORM_RE.search(l):
            return None
        srcs = OPERAND_RE.findall(l.split("=", 1)[1].split(":")[0])
        if not srcs:
            return None
        name = srcs[0]
    return None


num_join = 0
for op in ops:
    if not op["is_pb"]:
        continue
    work, seen = [op["result"]], set()
    while work:
        v = work.pop()
        if v in seen:
            continue
        seen.add(v)
        for u in users_of(v):
            ul = lines[u]
            if root_of(ul) != op["root"]:
                continue  # Outside_User
            uop = line_to_op.get(u)
            if uop is not None and uop["is_pb"]:
                work.append(uop["result"])
                continue
            if uop is None and DIM_TRANSFORM_RE.search(ul):
                r = re.match(r"^\s*(%[\w]+)", ul)
                work.append(r.group(1))
                continue
            if uop is not None and v not in uop["ins"]:
                continue  # read as an init only
            if uop is None:
                violations.append(
                    "line %d: broadcast %s reaches non-generic op at line %d "
                    "before the multiply" % (op["line"], op["result"], u + 1))
                continue
            direct = by_result.get(v)
            delta = inserted_loop_dim(direct, uop) if direct is not None \
                and direct["is_pb"] else None
            ok = False
            for x in uop["ins"]:
                if x == v:
                    continue
                spb = source_pb(x)
                if spb is None:
                    ok = True
                elif delta is None or inserted_loop_dim(spb, uop) != delta:
                    ok = True
            if ok:
                num_join += 1
            else:
                violations.append(
                    "line %d: broadcast %s reaches %s (line %d), which computes "
                    "on broadcast data before the multiply" %
                    (op["line"], op["result"], uop["result"], uop["line"]))

for v in violations:
    print("VIOLATION: " + v)
print("PRE-INVARIANT %s %d %d" % ("FAIL" if violations else "OK", len(ops), num_pb))
sys.exit(1 if violations else 0)
