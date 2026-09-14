# Setting up on a new machine

## 1. Clone

```
git clone https://github.com/4oeuhtns/classiq-challenge-2026.git
cd classiq-challenge-2026
```

## 2. Python 3.13 — not 3.14

This is the one step that will waste an afternoon if you skip it. Every
published `classiq` release declares `Requires-Python >=3.10,<3.14`. On a 3.14
interpreter `pip install classiq` reports

```
ERROR: No matching distribution found for classiq
```

which reads like a missing package but is a version wall. There is no 3.14
build to find.

```
conda create -n classiq python=3.13 -y
conda activate classiq
pip install -r requirements.txt
```

Keep this separate from any 3.14 environment you use for other work rather than
downgrading that one in place.

## 3. Authenticate

```
python -c "import classiq; classiq.authenticate()"
```

Opens a browser. The token is stored outside the repo and is not committed.

## 4. Check it works

Offline first — no cloud calls, nothing to pay for:

```
python testing/test_terms.py          # 85 checks, includes the fits() acceptance test
python experiments/handbuilt.py       # exact phase check + depth/cx
```

`handbuilt.py` should print `phase error 1.6e-17`, `ancilla leak 0.0e+00`,
`depth 1906  cx 1251`. It needs only qiskit, so it also works in an environment
without the SDK.

Then the cloud path:

```
python experiments/check_handbuilt.py --verify
```

## 5. The cache (optional)

`cache/` is gitignored. It holds 375 synthesised QASM files keyed by
`cache_key()`, which hashes `classiq.__version__` and `COMPILER_VERSION`
alongside the design — so those entries only resolve under classiq 1.28.0,
which is what `requirements.txt` pins.

To carry it across, from the old machine:

```
tar -czf cache.tgz cache/        # 72 MB -> 4 MB
```

and unpack it in the repo root on the new one. Skipping this costs nothing but
re-synthesis time, and only for designs that were already measured. The current
gate-level work (`experiments/handbuilt.py`) never touches the cache: its score
comes from `qasm_metrics`, which is pure regex over the QASM.

## What is and isn't in the repo

| | |
|---|---|
| `logs/experiments.jsonl` | **committed.** 595 rows, ~6.6 h of cloud time, the input to `fit_proxy.py` and the `fits()` acceptance test. Irreplaceable. |
| `cache/` | ignored. Derived, SDK-version-keyed, grows without bound. |
| `__pycache__/`, `.ipynb_checkpoints/` | ignored. |
| auth tokens | never in the repo. |
