# Checks Property 7 (compute core preservation, Requirement 5.6).
#
# Usage: compute_core_check.py <recognize-only output> <coupling output>
#
# The compute core of each root is taken as: every linalg.reduce carrying
# qgemm.node of that root (the downstream reductions and the final
# reduction, which carries qgemm.root), plus the op producing each reduce's
# input (the multiply for the first reduction, the scaling multiply for the
# later ones). These ops all live in V_m ∩ V_n. For each of them the checker
# records its kind and result type (plus operand type and reduced dimensions
# for reduces; producer operand types may legitimately change when a broadcast
# is folded into or out of it) in program
# order and requires the two files to agree exactly per root, so m and n stay
# at the positions they had in the separated input. It also checks that the
# final reduction's result has the m/n extents at the m_dim/n_dim recorded in
# qgemm.root, read from the reference.
#
# In addition, every op with stage `compute` of the reference that is not a
# Pure_Broadcast is compared with the output by its full printed text: op
# kind, indexing maps (aliases expanded), iterator types, attributes
# including qgemm.node, body and types, with SSA names renumbered in order of
# appearance (Req 12.15). Indexing maps and types are compared exactly: a
# lift never changes a consumer's operand maps or types (Req 6.8). The one
# exception is a linalg.generic input whose broadcast Separation split out
# (Req 2.1-2.4): its reference map is not the identity and its output map is,
# and that input's map and type are not compared. The reference op must
# occur in the output (as a
# multiset per function). Under `--mode=pre` this covers the reference compute
# ops in V_m ∩ V_n only: the recognizer's compute stage starts at the first
# broadcast edge, so it may contain ops that vary along one parallel dim only
# (the Q4_K a_q block sum and the init expand_shape of the k-reduction), which
# pre legitimately rebuilds without the other dim (Req 5.1, 10.3). V_d is
# derived from the reference: a value varies along d if it carries d and is
# not produced by a Pure_Broadcast inserting d, an init or a constant, and an
# op varies along d if one of its data inputs does. Under `--mode=on-the-fly`
# no op loses an axis, so every reference compute op is compared.
#
# Usage: compute_core_check.py [--mode=pre|on-the-fly] <ref> <out>
# Prints "COMPUTE-CORE OK <roots> <ops> <compared> <exempt>" or one
# "VIOLATION: ..." per failure.
import os
import re
import sys
from collections import Counter

sys.dont_write_bytecode = True  # keep the source tree clean
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_check import analyze  # noqa: E402

MODE = "pre"
for a in list(sys.argv[1:]):
    if a.startswith("--mode="):
        MODE = a.split("=", 1)[1]
        sys.argv.remove(a)
if MODE not in ("pre", "on-the-fly") or len(sys.argv) != 3:
    print("usage: compute_core_check.py [--mode=pre|on-the-fly] REF OUT")
    sys.exit(2)

OP_RE = re.compile(r"^\s*(%[\w.]+)(?::\d+)? = (linalg\.\w+|arith\.\w+|tensor\.\w+)")
NODE_RE = re.compile(r"qgemm\.node = \{root = (\d+)")
ROOT_RE = re.compile(r"qgemm\.root = \{m_dim = (\d+) : i64, n_dim = (\d+)")
INS_RE = re.compile(r"ins\(([^:)]*):([^)]*)\)")
OUTS_RE = re.compile(r"outs\(([^:)]*):([^)]*)\)")
DIMS_RE = re.compile(r"dimensions = \[([^\]]*)\]")


def parse(path):
    ops = {}  # ssa name -> dict
    order = []
    with open(path) as f:
        lines = f.read().splitlines()
    func = None
    for l in lines:
        fm = re.match(r"^\s*func\.func @(\w+)", l)
        if fm:
            func = fm.group(1)
            continue
        m = OP_RE.match(l)
        if not m:
            continue
        name = (func, m.group(1))
        ins = INS_RE.search(l)
        outs = OUTS_RE.search(l)
        node = NODE_RE.search(l)
        root = ROOT_RE.search(l)
        dims = DIMS_RE.search(l)
        op = {
            "kind": m.group(2),
            "ins": [s.strip() for s in ins.group(1).split(",")] if ins else [],
            "in_types": ins.group(2).strip() if ins else "",
            "out_type": outs.group(2).strip() if outs else "",
            "root": (func, int(node.group(1))) if node else None,
            "mn": (int(root.group(1)), int(root.group(2))) if root else None,
            "dims": dims.group(1) if dims else "",
            "name": m.group(1),
        }
        ops[name] = op
        order.append(name)
    return ops, order


def core(path):
    ops, order = parse(path)
    per_root = {}
    finals = {}
    for name in order:
        op = ops[name]
        if op["kind"] != "linalg.reduce" or op["root"] is None:
            continue
        sig = per_root.setdefault(op["root"], [])
        if op["ins"]:
            prod = ops.get((name[0], op["ins"][0]))
            if prod is not None:
                sig.append(("producer", prod["kind"], prod["out_type"]))
        sig.append(("reduce", op["in_types"], op["out_type"], op["dims"]))
        if op["mn"] is not None:
            finals[op["root"]] = (op["mn"], op["out_type"])
    return per_root, finals


def shape(t):
    m = re.match(r"tensor<([\dx]+)x\w+>", t)
    return [int(x) for x in m.group(1).split("x")] if m else None


ref, ref_finals = core(sys.argv[1])
out, out_finals = core(sys.argv[2])
bad = 0
for r in sorted(set(ref) | set(out)):
    a, b = ref.get(r, []), out.get(r, [])
    if a != b:
        bad += 1
        print(f"VIOLATION: root {r}: compute core differs")
        for x, y in zip(a + [None] * len(b), b + [None] * len(a)):
            if x != y:
                print(f"  ref: {x}\n  out: {y}")
                break
for r, (mn, t) in ref_finals.items():
    o = out_finals.get(r)
    if o is None or o[0] != mn or o[1] != t:
        bad += 1
        print(f"VIOLATION: root {r}: final reduction {o} != {(mn, t)}")
if not ref:
    bad += 1
    print("VIOLATION: no root found in reference")


def expand_aliases(text):
    defs = dict(re.findall(r"^(#\w+) = (affine_map<.*>)$", text, re.M))
    return lambda l: re.sub(r"#\w+\b", lambda m: defs.get(m.group(0), m.group(0)), l)


def op_text(lines, i):
    """The printed text of the op starting at line i, including its body:
    a generic's region ends at its `} -> type` line, a reduce's region opens
    on the next line and ends at the matching `}`."""
    out = [lines[i]]
    j = i + 1
    if "= linalg.generic" in lines[i] and lines[i].rstrip().endswith("{"):
        while not re.match(r"^\s*\} ->", lines[j]):
            out.append(lines[j])
            j += 1
        out.append(lines[j])
    elif "= linalg.reduce" in lines[i] and j < len(lines) and \
            re.match(r"^\s*\(.*\)\s*\{\s*$", lines[j]):
        while not re.match(r"^\s*\}\s*$", lines[j]):
            out.append(lines[j])
            j += 1
        out.append(lines[j])
    return out


def split_top(s):
    """Split `s` at commas that are not nested in <>, () or []."""
    parts, depth, cur = [], 0, ""
    for k, c in enumerate(s):
        if c in "<([":
            depth += 1
        elif c in ">)]" and not (c == ">" and k and s[k - 1] == "-"):
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        cur += c
    if cur.strip():
        parts.append(cur.strip())
    return parts
MAPS_RE = re.compile(r"indexing_maps = \[(.*?)\], iterator_types")
GEN_INS_RE = re.compile(r"ins\(([^:)]*) : ([^)]*)\)")


def is_identity(m):
    rm = re.match(r"affine_map<\((.*?)\) -> \((.*)\)>", m)
    return rm is not None and rm.group(1) == rm.group(2)


def split_inputs(head):
    """Split a linalg.generic header into the header without its input maps
    and types, and the list of (map, type) of its inputs."""
    if "linalg.generic" not in head:
        return head, ()
    mm, im = MAPS_RE.search(head), GEN_INS_RE.search(head)
    if not mm or not im:
        return head, ()
    maps = split_top(mm.group(1))
    types = split_top(im.group(2))
    inputs = tuple(zip(maps, types))
    head = head.replace(mm.group(1), ", ".join(maps[len(types):]), 1)
    return head.replace(im.group(0), "ins(<%d>)" % len(types), 1), inputs


def matches(ref, out):
    """`out` equals `ref` except for inputs whose broadcast Separation split
    out: non-identity map in the reference, identity map in the output."""
    if ref[0] != out[0] or ref[2] != out[2] or len(ref[1]) != len(out[1]):
        return False
    return all(r == o or (not is_identity(r[0]) and is_identity(o[0]))
               for r, o in zip(ref[1], out[1]))


def signature(lines, i, exp):
    """Op text with aliases expanded. The header's operand and result names
    become %_ (which value feeds an op is not part of its own signature, and
    init sharing is input-dependent, Req 6.8); names inside the body are
    renumbered in order of appearance, so the body's dataflow is compared."""
    names = {}

    def ren(m):
        names.setdefault(m.group(0), "%%v%d" % len(names))
        return names[m.group(0)]
    text = [re.sub(r"\s+loc\(.*?\)", "", exp(l).strip())
            for l in op_text(lines, i)]
    head = re.sub(r"%[\w.$-]+(:\d+|#\d+)?", "%_", text[0])
    head, inputs = split_inputs(head)
    body = [re.sub(r"%[\w.$-]+", ren, l) for l in text[1:]]
    return (head, inputs, "\n".join(body))


def is_pure_broadcast(op):
    if op["kind"] != "generic" or len(op["ins"]) != 1:
        return False
    dims, out = op["maps"][-1]
    kept = op["maps"][0][1]
    return (out == dims and len(op["body"]) == 1 and
            re.fullmatch(r"linalg\.yield %in : \S+", op["body"][0]) is not None
            and any(op["labels"].get(p) for p, d in enumerate(dims)
                    if d not in kept))


def compute_ops(path, mode):
    text = open(path).read()
    lines, ops, labels = analyze(text)
    exp = expand_aliases(text)
    by_res = {op["result"]: op for op in ops if op["result"]}
    defline = {}
    for i, l in enumerate(lines):
        m = re.match(r"^\s*(%[\w]+)(?::\d+)? = (\S+)", l)
        if m:
            defline[m.group(1)] = m.group(2)
    vary = {}

    def varies(v):
        # Set of parallel dims along which value v varies.
        if v in vary:
            return vary[v]
        vary[v] = set()
        op = by_res.get(v)
        # An annotated op without tensor inputs is a leaf of the dataflow as
        # much as an unannotated one: the operand load chain is annotated now,
        # so bufferization.to_tensor is an annotated leaf whose axes are read
        # from its own labels rather than from inputs it does not have.
        if op is None or not op.get("ins"):
            kind = defline.get(v, "")
            if re.match(r"(tensor\.empty|linalg\.fill|arith\.constant)", kind):
                res = set()
            else:
                res = set(labels.get(v, {}).values())
        else:
            res = set()
            for x in op.get("ins", []):
                res |= varies(x)
            if is_pure_broadcast(op):
                dims, out = op["maps"][-1]
                kept = op["maps"][0][1]
                ins_d = {op["labels"].get(p) for p, d in enumerate(dims)
                         if d not in kept}
                res -= ins_d
            res &= set(op["labels"].values())
        vary[v] = res
        return res

    func = None
    funcs = {}
    for i, l in enumerate(lines):
        fm = re.match(r"^\s*func\.func @(\w+)", l)
        if fm:
            func = fm.group(1)
        funcs[i] = func
    sigs, exempt = Counter(), 0
    for op in ops:
        if op["stage"] != "compute" or is_pure_broadcast(op):
            continue
        if mode == "pre" and op["result"] and \
                varies(op["result"]) != {"m", "n"}:
            exempt += 1
            continue
        sigs[(funcs[op["line"]], signature(lines, op["line"], exp))] += 1
    return sigs, exempt


def all_ops(path):
    text = open(path).read()
    lines = text.splitlines()
    exp = expand_aliases(text)
    out = Counter()
    func = None
    for i, l in enumerate(lines):
        fm = re.match(r"^\s*func\.func @(\w+)", l)
        if fm:
            func = fm.group(1)
        if re.match(r"^\s*%[\w]+(?::\d+)? = ", l):
            out[(func, signature(lines, i, exp))] += 1
    return out


def show(sig):
    head, inputs, body = sig
    text = head + "".join("\n  in: %s : %s" % i for i in inputs)
    return text + ("\n" + body if body else "")


ref_sigs, n_exempt = compute_ops(sys.argv[1], MODE)
out_sigs = all_ops(sys.argv[2])
# Multiset matching: exact matches first, then the Separation exception.
for (f, sig), k in list(ref_sigs.items()):
    take = min(k, out_sigs[(f, sig)])
    out_sigs[(f, sig)] -= take
    k -= take
    for key in list(out_sigs):
        if k == 0:
            break
        if key[0] != f or out_sigs[key] == 0 or not matches(sig, key[1]):
            continue
        take = min(k, out_sigs[key])
        out_sigs[key] -= take
        k -= take
    if k:
        bad += 1
        print("VIOLATION: @%s: reference compute op not in output (x%d):"
              % (f, k))
        print("  " + show(sig).replace("\n", "\n  "))
if not ref_sigs:
    bad += 1
    print("VIOLATION: no compute op compared")

if bad:
    sys.exit(1)
print(f"COMPUTE-CORE OK {len(ref)} {sum(len(v) for v in ref.values())} "
      f"{sum(ref_sigs.values())} {n_exempt}")
