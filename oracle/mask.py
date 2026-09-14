"""The target image: what the phase oracle must mark.

Masks are GRID_SIZE x GRID_SIZE boolean arrays indexed ``mask[y][x]`` -- the
first index is the row (y), the second is the column (x). Nothing is ever
flipped in storage; ``draw`` handles orientation at display time with
``origin="lower"``, so y=0 appears as the bottom row on screen.

This is the bottom layer of the package. It imports nothing else from
``oracle``, and every other module takes its notion of "the grid" from here.
"""

import numpy as np

COORD_BITS = 6
"""Bits per coordinate axis. x and y each occupy this many qubits."""

GRID_SIZE = 1 << COORD_BITS  # 2^6 = 64
"""Width and height of the image in pixels. Always ``2 ** COORD_BITS``."""


def logo_pixel(x: int, y: int) -> bool:
    """Return True exactly for black pixel (x, y).

    The challenge specification, copied verbatim from the baseline notebook,
    and the single source of truth for the target image. Everything else is
    derived from it -- never restate this geometry elsewhere, or the two copies
    will silently drift apart.

    The logo is the union of four shapes:

    - stem: 25 x 25 block, x in [2, 26], y in [29, 53]
    - arm: 24 x 5 block, x in [26, 49], y in [39, 43]
    - small disk: radius sqrt(42) about (55, 41)
    - large disk: radius sqrt(72) about (40, 19)

    Args:
        x: Column, 0 to GRID_SIZE - 1.
        y: Row, 0 to GRID_SIZE - 1.

    Returns:
        True if the pixel is black, i.e. must receive a -1 phase.
    """
    return (
        (2 <= x <= 26 and 29 <= y <= 53)
        or (26 <= x <= 49 and 39 <= y <= 43)
        or (x - 55) ** 2 + (y - 41) ** 2 <= 42
        or (x - 40) ** 2 + (y - 19) ** 2 <= 72
    )


TARGET = np.array(
    [[logo_pixel(x, y) for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]
)
"""The goal image as a boolean mask, indexed ``TARGET[y][x]``.

Exactly 1097 pixels are True. A design is correct when the XOR of its rendered
terms equals this array -- see ``oracle.terms.matches_target``.
"""

assert TARGET.sum() == 1097, f"target must have 1097 black pixels, got {TARGET.sum()}"


def draw(mask):
    """Display a mask as an image, with y increasing upwards.

    Takes a mask rather than a term, so the same function serves every use:
    one term's ``render()``, a whole design's ``cover_mask()``, ``TARGET``, or
    the XOR of two of those -- the error map, which is the plot worth staring
    at, since it is True exactly where the design is still wrong.

    Args:
        mask: A GRID_SIZE x GRID_SIZE boolean array indexed [y][x].
    """
    import matplotlib.patches as patches  # TODO: outlines for different terms
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.imshow(
        mask,
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
        extent=(0, GRID_SIZE, 0, GRID_SIZE),
    )
    plt.show()
