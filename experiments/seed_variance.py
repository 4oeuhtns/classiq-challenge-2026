import pathlib
import sys

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

from oracle.evaluate import evaluate
from oracle.baseline import BASELINE_TERMS

depths = []
cxs = []
for i in range(20):
    m, v, run_id = evaluate(BASELINE_TERMS, seed=i, with_verify=False, strategy_tags=["seed_variance_test"])
    if m is not None:
        depths.append(m.depth)
        cxs.append(m.cx)
        print(f"Seed: {i}, Depth: {m.depth}, Cx: {m.cx}")

diff_d = set(depths)
print(diff_d)
if depths:
    max_depth = max(depths)
    min_depth = min(depths)
    print(f"max depth: {max_depth}")
    print(f"min depth: {min_depth}")

diff_c = set(cxs)
print(diff_c)
if cxs:
    max_cx = max(cxs)
    min_cx = min(cxs)
    print(f"max cx: {max_cx}")
    print(f"min cx: {min_cx}")


