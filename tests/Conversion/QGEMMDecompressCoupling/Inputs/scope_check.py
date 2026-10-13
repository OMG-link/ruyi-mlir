# Checks scope preservation (Property 11) and broadcast uniqueness
# (Property 12) of --qgemm-decompress-coupling.
#
# Usage: scope_check.py <recognize-only output> < <pass output>
# Both inputs must be printed with --mlir-print-op-generic
# --mlir-print-local-scope, so every op carries its full type signature and
# inline affine maps.
#
# Verified, per func.func:
#  1. Ops outside every scope (no qgemm.node anywhere in the op's text) other
#     than tensor.empty, linalg.fill and constants form the same multiset in
#     both inputs, compared on the op text with SSA names and block labels
#     erased: kind, operand types, result types, properties, attributes and
#     regions. Outside_Users (unannotated ops and the terminator) are
#     included, so their operand types are checked as well.
#  2. In the pass output, no tensor.empty, linalg.fill or constant, and no
#     annotated op, has a result without users.
#  3. In the pass output, no two annotated Pure_Broadcasts (single-input
#     linalg.generic yielding its input unchanged, or linalg.broadcast) share
#     the same input value and the same input indexing map / dimensions.
# Prints "SCOPE OK <n-outside> <n-pb>" or one "VIOLATION: ..." per failure.
import collections
import re
import sys

NEUTRAL = ("tensor.empty", "linalg.fill", "arith.constant")


def top_level_ops(text):
    """Returns {func name: [op text block]} for the ops directly in each
    func.func body (generic form: func at indent 2, its ops at indent 4)."""
    funcs = {}
    cur = None
    block = None
    for l in text.splitlines():
        ind = len(l) - len(l.lstrip(" "))
        if ind == 2 and l.lstrip().startswith("^"):
            continue
        if ind == 2 and '"func.func"' in l:
            m = re.search(r'sym_name = "([\w]+)"', l)
            cur = funcs.setdefault(m.group(1), [])
            block = None
            continue
        if ind == 2 and cur is not None:
            cur = None
            continue
        if cur is None:
            continue
        if ind == 4 and not l.lstrip().startswith(("^", "}")):
            block = [l]
            cur.append(block)
        elif ind >= 4 and block is not None:
            block.append(l)
    return {f: ["\n".join(b) for b in bs] for f, bs in funcs.items()}


def kind(op):
    return re.search(r'"([\w.]+)"\(', op).group(1)


def results(op):
    m = re.match(r"\s*(%[\w]+)(?::(\d+))? = ", op)
    if not m:
        return []
    return [m.group(1)]


def normalize(op):
    op = re.sub(r"%[\w]+(#\d+)?(:\d+)?", "%_", op)
    return re.sub(r"\^bb\d+", "^bb", op).strip()


def is_neutral(op):
    k = kind(op)
    return k in NEUTRAL or k.endswith(".constant")


ref = top_level_ops(open(sys.argv[1]).read())
out_text = sys.stdin.read()
out = top_level_ops(out_text)
violations = []
n_outside = 0
n_pb = 0
if set(ref) != set(out):
    violations.append("function set differs: %s vs %s" % (sorted(ref), sorted(out)))
for f in sorted(set(ref) & set(out)):
    def outside(ops):
        return collections.Counter(normalize(o) for o in ops
                                   if "qgemm.node" not in o and not is_neutral(o))
    a, b = outside(ref[f]), outside(out[f])
    n_outside += sum(b.values())
    for o in (a - b):
        violations.append("@%s: outside op lost or changed: %s" % (f, o.splitlines()[0]))
    for o in (b - a):
        violations.append("@%s: outside op added or changed: %s" % (f, o.splitlines()[0]))
    body = "\n".join(out[f])
    for o in out[f]:
        if not (is_neutral(o) or "qgemm.node" in o):
            continue
        for r in results(o):
            uses = re.findall(re.escape(r) + r"(?![\w])", body)
            if len(uses) <= 1:
                violations.append("@%s: zero-user %s %s" % (f, kind(o), r))
    seen = {}
    for o in out[f]:
        if "qgemm.node" not in o:
            continue
        key = None
        k = kind(o)
        if k == "linalg.broadcast":
            m = re.search(r'\((%[\w#]+), %[\w#]+\)', o)
            d = re.search(r"dimensions = array<i64: ([^>]*)>", o)
            key = (m.group(1), d.group(1))
        elif k == "linalg.generic":
            ins = re.match(r'\s*%[\w]+ = "linalg.generic"\(([^)]*)\)', o)
            ops_ = [x.strip() for x in ins.group(1).split(",")]
            seg = re.search(r"operandSegmentSizes = array<i32: (\d+), (\d+)>", o)
            lines = [l.strip() for l in o.splitlines()[1:]
                     if l.strip() and not l.strip().startswith("^")
                     and not l.strip().startswith("})")]
            bb = re.search(r"\^bb0\((%[\w]+):", o)
            if seg and seg.group(1) == "1" and len(lines) == 1 and bb and \
                    lines[0].startswith('"linalg.yield"(%s)' % bb.group(1)):
                maps = re.search(r"indexing_maps = \[(.*?)\]", o).group(1)
                in_map = re.split(r",\s*(?=affine_map)", maps)[0]
                key = (ops_[0], in_map)
        if key is None:
            continue
        n_pb += 1
        if key in seen:
            violations.append("@%s: duplicate broadcast of %s along %s" % (f, key[0], key[1]))
        seen[key] = True
for v in violations:
    print("VIOLATION: " + v)
print("SCOPE %s %d %d" % ("FAIL" if violations else "OK", n_outside, n_pb))
sys.exit(1 if violations else 0)
