"""Shared batch loop for the experiment scripts.

Both `calibrate.py` and `sweep.py` need the same four things: resume after a
crash, stop at a deadline, survive a bad candidate, and show progress. This is
that loop, in one place, so a fix to it can't reach one script and miss the
other.

It writes nothing itself -- every result reaches disk through `evaluate`,
which appends a row to logs/experiments.jsonl and caches the QASM.

**Failures never stop the batch.** `evaluate` already turns a width refusal
into a logged row, and the broad `except` here catches everything else. That
is the opposite of the rule inside `evaluate`, where a TypeError is a bug and
must crash. Different layer, different rule: one malformed candidate must not
cost you the night.
"""

import pathlib
import sys
import time

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

from oracle.log import rows
from oracle.terms import fingerprint


def hhmm(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def refused_before():
    """Fingerprints of designs the server refused on width.

    Only genuine width refusals, identified by `required_width` being set. A
    design that wants 38 qubits will want 38 again, so retrying costs 30 s to
    be told the same thing.

    A row with an `error` but no `required_width` is something else -- a 502,
    a dropped connection -- and those are transient. Treating them as
    permanent would blacklist a perfectly good design forever.
    """
    return {r["cover_fingerprint"] for r in rows() if r.get("required_width") is not None}


def measured_before():
    """Fingerprints that already have a depth."""
    return {r["cover_fingerprint"] for r in rows() if r.get("real_depth") is not None}


def run_batch(designs, hours=5.0, with_verify=False, seed=0, skip_measured=True):
    """Measure a list of designs, returning one result per design.

    Args:
        designs: list of (terms, tags). `tags` goes to the log's
            `strategy_tags`, so every row can be traced to what produced it.
        hours: stop cleanly after this long. The caller's designs should be
            ordered so that a truncated run is still useful.
        with_verify: also simulate each circuit. Roughly doubles the runtime,
            and is unnecessary per candidate -- verification is a check on the
            compiler, so it belongs per code path (see testing/test_verify.py).
        seed: synthesis seed. Measured 2026-09-13 to make no difference at all
            -- 19 real calls across 20 seeds returned byte-identical metrics --
            so one seed per measurement is the fixed policy.
        skip_measured: skip designs that already have a depth in the log.
            True for a long batch, where re-logging hundreds of duplicate rows
            is just noise. False for a small sweep, where you want the full
            table back even on a re-run (cache hits cost ~3 ms).

    Returns:
        List of (terms, tags, measurement) in the order given. `measurement`
        is None for a design that was refused, errored, or never attempted.
    """
    skip = refused_before() | (measured_before() if skip_measured else set())
    todo = [(t, tags) for t, tags in designs if fingerprint(t) not in skip]
    results = {}

    print(f"{len(designs)} designs, {len(designs) - len(todo)} already answered, "
          f"{len(todo)} to measure")
    if len(todo) > 50:
        print("\nPrevent the machine from sleeping, or this stops when the lid closes.")
        print("  Windows:  powercfg /change standby-timeout-ac 0")
        print("  restore:  powercfg /change standby-timeout-ac 30")
    print()

    # imported here so a caller that only wants the design list needs no SDK
    from oracle.evaluate import evaluate

    started = time.perf_counter()
    deadline = started + hours * 3600
    depths, failures, stopped = [], 0, None

    for i, (terms, tags) in enumerate(todo, 1):
        if time.perf_counter() > deadline:
            stopped = "deadline"
            break
        label = tags[-1] if tags else "?"
        try:
            m, _, _ = evaluate(terms, seed=seed, with_verify=with_verify, strategy_tags=tags)
        except KeyboardInterrupt:
            stopped = "interrupted"
            break
        except Exception as exc:                                  # noqa: BLE001
            failures += 1
            print(f"[{i:>3}/{len(todo)}] {label:<16} n={len(terms):<3} "
                  f"!! {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
            continue

        results[fingerprint(terms)] = m
        elapsed = time.perf_counter() - started
        if m is None:
            failures += 1
            outcome = "refused by the server"
        else:
            depths.append(m.depth)
            outcome = f"depth={m.depth:<6} cx={m.cx:<6}"
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i:>3}/{len(todo)}] {label:<16} n={len(terms):<3} {outcome:<24} "
              f"elapsed {hhmm(elapsed)}  eta {hhmm(eta)}")

    print()
    if stopped:
        print(f"stopped early: {stopped}. Re-run the same command to continue where it left off.")
    print(f"measured {len(depths)}, failed {failures}, in {hhmm(time.perf_counter() - started)}")
    if depths:
        depths.sort()
        print(f"depth range: {depths[0]} .. {depths[-1]}   median {depths[len(depths) // 2]}"
              f"   ({depths[-1] / max(depths[0], 1):.1f}x spread)")

    return [(t, tags, results.get(fingerprint(t))) for t, tags in designs]
