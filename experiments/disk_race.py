"""Test the skepticism directly: is a disk, built the same hand-written way as
the current best design (rank-1 rectangle pieces + shared-scratch MCX/MCZ,
not Classiq's synthesizer), actually cheap in isolation -- or does it blow
past the leaderboard's entire 137/561 budget on its own, the way N rectangles
synthesized through Classiq's automatic path have already been shown to?

This reuses handbuilt.py's build()/MCX/MCZ/best_blocks machinery UNCHANGED --
no new gate logic, just a different, isolated `pieces` list -- so correctness
rests on code that's already verified, not on anything new.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from handbuilt import build, score, X, Y, NB

GRID = 64


def disk1_pixel(x, y):
    return (x - 40) ** 2 + (y - 19) ** 2 <= 72


def disk2_pixel(x, y):
    return (x - 55) ** 2 + (y - 41) ** 2 <= 42


def row_intervals(pixel_fn, y):
    xs = [x for x in range(GRID) if pixel_fn(x, y)]
    if not xs:
        return ()
    out, start, end = [], xs[0], xs[0]
    for x in xs[1:]:
        if x == end + 1:
            end = x
        else:
            out.append((start, end))
            start = end = x
    out.append((start, end))
    return tuple(out)


def row_staircase_rects(pixel_fn):
    """Merge identical intervals on adjacent rows into rectangles -- same
    algorithm as oracle/baseline.py's merge_equal_intervals, generalized to
    any predicate instead of being hardcoded to the full logo."""
    active, rects = {}, []
    for y in range(GRID + 1):
        intervals = set(row_intervals(pixel_fn, y)) if y < GRID else set()
        for interval, y_min in tuple(active.items()):
            if interval not in intervals:
                rects.append((*interval, y_min, y - 1))
                del active[interval]
        for interval in intervals:
            active.setdefault(interval, y)
    return sorted(rects, key=lambda r: (r[2], r[0]))


def mask_of(lo, hi):
    return ((1 << (hi - lo + 1)) - 1) << lo


def rects_to_pieces(rects):
    return [(mask_of(y0, y1), mask_of(x0, x1)) for x0, x1, y0, y1 in rects]


def disk_target(pixel_fn):
    return np.array([[pixel_fn(x, y) for x in range(GRID)] for y in range(GRID)])


def gf2_rank_decompose(mask_2d):
    """The same technique that found rank=10 for the whole logo (a standard
    GF(2) linear/XOR basis), applied to just one mask. Returns (ymask, xmask)
    pieces -- rank-minimal, but NOT basis-searched for circuit cost the way
    handbuilt.py's actual PIECES were; a fair floor, not the true optimum."""
    rows = [sum(1 << x for x in range(GRID) if mask_2d[y][x]) for y in range(GRID)]

    basis = {}      # leading_bit -> (pattern, piece_index)
    xmasks = []
    for y in range(GRID):
        cur = rows[y]
        while cur:
            k = cur.bit_length() - 1
            if k not in basis:
                basis[k] = (cur, len(xmasks))
                xmasks.append(cur)
                cur = 0
            else:
                cur ^= basis[k][0]

    ymasks = [0] * len(xmasks)
    for y in range(GRID):
        cur = rows[y]
        while cur:
            k = cur.bit_length() - 1
            pat, idx = basis[k]
            ymasks[idx] |= (1 << y)
            cur ^= pat

    return list(zip(ymasks, xmasks))


def verify(qc, target):
    full = QuantumCircuit(18)
    for q in X + Y:
        full.h(q)
    full.compose(qc, inplace=True)
    amp = np.asarray(Statevector(full).data).reshape(64, 64, 64)
    want = np.where(target, -1.0, 1.0) / 64.0
    return float(np.abs(amp[0] - want).max()), float(np.abs(amp[1:]).max())


def report(label, pixel_fn, pieces):
    n_px = int(disk_target(pixel_fn).sum())
    qc = build(pieces=pieces)
    err, leak = verify(qc, disk_target(pixel_fn))
    d, cx = score(qc, tries=[(3, s) for s in range(8)] + [(2, 0)])
    print(f"{label}: {n_px} pixels, {len(pieces)} rectangle terms, "
          f"phase err {err:.1e}, leak {leak:.1e}")
    print(f"{label}: depth={d}  cx={cx}")
    return d, cx


if __name__ == "__main__":
    d1_rects = row_staircase_rects(disk1_pixel)
    d2_rects = row_staircase_rects(disk2_pixel)
    print(f"disk 1 row-staircase: {len(d1_rects)} rectangles -> {d1_rects}")
    print(f"disk 2 row-staircase: {len(d2_rects)} rectangles -> {d2_rects}\n")

    report("disk 1 alone", disk1_pixel, rects_to_pieces(d1_rects))
    print()
    report("disk 2 alone", disk2_pixel, rects_to_pieces(d2_rects))
    print()

    def both_pixel(x, y):
        return disk1_pixel(x, y) or disk2_pixel(x, y)

    both_pieces = rects_to_pieces(d1_rects) + rects_to_pieces(d2_rects)
    report("both disks combined (row-staircase)", both_pixel, both_pieces)

    print("\n--- now the same disks, GF(2) rank-decomposed instead of row-staircase ---\n")

    d1_rank_pieces = gf2_rank_decompose(disk_target(disk1_pixel))
    d2_rank_pieces = gf2_rank_decompose(disk_target(disk2_pixel))
    both_rank_pieces = gf2_rank_decompose(disk_target(both_pixel))
    print(f"disk 1 rank: {len(d1_rank_pieces)}   disk 2 rank: {len(d2_rank_pieces)}   "
          f"both (joint rank): {len(both_rank_pieces)}\n")

    report("disk 1 alone (rank)", disk1_pixel, d1_rank_pieces)
    print()
    report("disk 2 alone (rank)", disk2_pixel, d2_rank_pieces)
    print()
    report("both disks combined (rank)", both_pixel, both_rank_pieces)
