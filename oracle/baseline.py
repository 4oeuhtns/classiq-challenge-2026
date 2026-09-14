"""The baseline design: the reference every later result is measured against.

Ports the rectangle cover from the challenge's baseline notebook -- a greedy
row sweep that produces 18 pairwise-disjoint rectangles exactly covering the
logo. It is deliberately unoptimised. This is not a design to improve in
place; it is the number to beat.

Sits above ``terms`` in the layering: it needs ``logo_pixel`` from ``mask`` and
``Rect`` from ``terms``, so neither of those may ever import this module.

The asserts at the bottom run on import, which makes importing this module its
own test. To check it in isolation, from the project root::

    python -m oracle.baseline

Use ``-m``, not ``python oracle/baseline.py`` -- the relative imports on line 1
only resolve when the file is loaded as part of the ``oracle`` package.
"""

from .mask import GRID_SIZE, logo_pixel
from .terms import Rect, matches_target, cover_mask


def intervals_for_row(y: int) -> tuple[tuple[int, int], ...]:
    """Return the contiguous black-pixel intervals in row y.

    Args:
        y: Row index, 0 to GRID_SIZE - 1.

    Returns:
        (x_start, x_end) pairs with inclusive bounds, ordered left to right.
        An empty tuple for a row containing no black pixels.
    """
    xs = [x for x in range(GRID_SIZE) if logo_pixel(x, y)]
    if not xs:
        return ()

    intervals: list[tuple[int, int]] = []
    start = xs[0]
    end = xs[0]
    for x in xs[1:]:
        if x == end + 1:
            end = x
        else:
            intervals.append((start, end))
            start = x
            end = x
    intervals.append((start, end))
    return tuple(intervals)


def merge_equal_intervals() -> tuple[tuple[int, int, int, int], ...]:
    """Merge equal intervals on adjacent rows into disjoint rectangles.

    Sweeps y upward. ``active`` maps each interval currently being extended to
    the row it started on. An interval that fails to reappear in the next row
    is closed off into a finished rectangle. The loop deliberately runs one row
    past the top of the grid -- a sentinel row with no intervals -- so that
    anything still open at y = GRID_SIZE - 1 gets closed too.

    Because rows only ever merge with an *identical* interval, the result is
    pairwise disjoint: this is a union cover, not a parity cover, and it uses
    none of the cancellation that ``cover_mask`` allows.

    Returns:
        (x_min, x_max, y_min, y_max) tuples, all bounds inclusive, sorted by
        y_min then x_min. This field order matches Rect's exactly. Exactly 18
        for this logo.
    """
    active: dict[tuple[int, int], int] = {}
    rectangles: list[tuple[int, int, int, int]] = []

    for y in range(GRID_SIZE + 1):
        intervals = set(intervals_for_row(y)) if y < GRID_SIZE else set()

        for interval, y_min in tuple(active.items()):
            if interval not in intervals:
                rectangles.append((*interval, y_min, y - 1))
                del active[interval]

        for interval in intervals:
            active.setdefault(interval, y)

    return tuple(sorted(rectangles, key=lambda block: (block[2], block[0])))


ORACLE_RECTANGLES = merge_equal_intervals()
"""The baseline cover as raw (x_min, x_max, y_min, y_max) tuples.

Kept alongside BASELINE_TERMS because Part III of the plan reasons about these
coordinates directly -- see III.7 on the mirrored x-ranges in the disks.
"""

BASELINE_TERMS = tuple([Rect(*t) for t in ORACLE_RECTANGLES])
"""The baseline design, as a tuple of 18 Rect terms.

A tuple, not a list, so an in-place mutation by a caller cannot rewrite the
reference design -- and because `fingerprint` is deliberately order-sensitive,
such a mutation would silently change the cache key too.

The reference point for the whole project: ``measure(BASELINE_TERMS)`` gives the
depth that every candidate design has to beat.
"""

assert len(BASELINE_TERMS) == 18, f"length of BASELINE_TERMS {len(BASELINE_TERMS)}, want 18"
assert matches_target(BASELINE_TERMS), f"{cover_mask(BASELINE_TERMS).sum()} pixels covered, want 1097"
