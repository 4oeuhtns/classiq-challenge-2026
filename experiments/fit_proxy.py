import pathlib
import sys

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

from scipy.stats import spearmanr, rankdata
import collections

from oracle.log import rows
from oracle.terms import deserialize
from oracle.proxy import PROXIES, features

def load_terms():
    experiments = rows()
    fingerprint_set = set()
    data = []
    for e in experiments:
        if e["real_depth"] is None:
            continue
        if e["cover_fingerprint"] in fingerprint_set:
            continue
        # filter sweeping and probing runs
        tag = (e["strategy_tags"] or ["?"])[0]
        if tag in ("sweep", "probe"):
            continue
        
        data.append((deserialize(e["terms_json"]), e["real_depth"], (e["strategy_tags"] or ["?"])[0]))
        fingerprint_set.add(e["cover_fingerprint"])
        
    return data

data = load_terms()
feats  = [features(t) for t, d, r in data]
actual = [d for t, d, r in data]
actual_rank = rankdata(actual)

for name, fn in PROXIES.items():
    pred = [fn(f) for f in feats]
    print(f"{name}  rho = {spearmanr(pred, actual).statistic:+.3f}   (n={len(data)})")

    pred_rank   = rankdata(pred)
    err = pred_rank - actual_rank
    worst = sorted(range(len(data)), key=lambda i: -abs(err[i]))[:10]
    for i in worst:
        t, d, recipe = data[i]
        print(f"{recipe:<14} n={len(t):<3} controls={sum(feats[i]['controls']):<4} "
            f"depth={d:<6} pred_rank={pred_rank[i]:6.0f} actual_rank={actual_rank[i]:6.0f} "
            f"err={err[i]:+7.0f}")

    by = collections.defaultdict(list)
    for i, (t, d, recipe) in enumerate(data):
        by[recipe].append(err[i])

    for recipe, errs in sorted(by.items(), key=lambda kv: -abs(sum(kv[1]) / len(kv[1]))):
        print(f"{recipe:<14} n={len(errs):<3} mean rank error {sum(errs) / len(errs):+7.1f}")