"""Term types: the design, expressed as data rather than code.

A *term* is any object with a ``render()`` method returning a
GRID_SIZE x GRID_SIZE boolean mask. A *design* is a plain list of terms.

Terms combine by XOR, not union: a pixel is black when an odd number of terms
cover it. Covers may therefore overlap freely, and stamping a region twice
erases it. The baseline's disjoint rectangles are a self-imposed restriction,
not a rule of the challenge.

Every term is a frozen dataclass, so terms are immutable, hashable, and
compared by value. That is what lets a design serve as a cache key and be
mutated cheaply by the search later on.

Coordinates here are per-axis: x and y each run 0 to GRID_SIZE - 1, and Cube
bit indices run 0 to COORD_BITS - 1 with bit 0 worth 1. Mapping those onto the
12-qubit register is compile.py's job, never this module's.
"""

import numpy as np
from dataclasses import dataclass, asdict, is_dataclass, fields
import hashlib
import json

from .mask import GRID_SIZE, TARGET

from typing import Protocol

class Term(Protocol):
    def render(self) -> np.ndarray:
        return None


@dataclass(frozen=True)
class Rect:
    """An axis-aligned block of pixels, with inclusive bounds.

    Both ends are included, so ``Rect(2, 26, 29, 53)`` spans 25 columns and
    25 rows -- 625 pixels, not 576.

    Attributes:
        min_x: Leftmost column, inclusive.
        max_x: Rightmost column, inclusive.
        min_y: Bottom row, inclusive.
        max_y: Top row, inclusive.
    """

    min_x: int
    max_x: int
    min_y: int
    max_y: int

    def render(self) -> np.ndarray:
        """Return this rectangle as a mask.

        Returns:
            A GRID_SIZE x GRID_SIZE boolean array indexed [y][x], True inside
            the rectangle and False everywhere else.
        """
        vals = np.arange(GRID_SIZE)  # [0, 1, 2, ..., GRID_SIZE-1]

        in_x = (vals >= self.min_x) & (vals <= self.max_x)  # x row where True if within range
        in_y = (vals >= self.min_y) & (vals <= self.max_y)  # y column

        return in_x[None, :] & in_y[:, None]  # combines in_x row (shape(1, n)) and in_y column (shape(n, 1))


@dataclass(frozen=True)
class Cube:
    """A set of pixels defined by fixing individual coordinate bits.

    Unfixed bits stay free, so a Cube covers every pixel whose fixed bits
    match. This expresses patterns no rectangle can: fixing only y bit 4
    selects y in [16, 31] *and* y in [48, 63] -- two stripes from one term.

    Bit indices are little-endian and per-axis. Bit 0 is worth 1, bit 5 is
    worth 32. They are not qubit numbers.

    Build these with ``Cube.fix``, which takes dicts and normalises them.

    Attributes:
        x_bits: Sorted (bit_index, value) pairs constraining x.
        y_bits: Sorted (bit_index, value) pairs constraining y.
    """

    x_bits: tuple[tuple[int, int], ...]
    y_bits: tuple[tuple[int, int], ...]

    @classmethod
    def fix(cls, x=None, y=None) -> "Cube":
        """Build a Cube from dicts of ``{bit_index: value}``.

        Sorting the pairs makes the stored form canonical, so two Cubes
        describing the same pixels compare equal and hash alike::

            Cube.fix(x={5: 1, 4: 0}) == Cube.fix(x={4: 0, 5: 1})

        Args:
            x: Bits to fix on the x axis, e.g. ``{5: 1, 4: 0}``. None fixes none.
            y: Bits to fix on the y axis. None fixes none.

        Returns:
            A new Cube. With no arguments, one covering the whole grid.
        """
        # works if x or y are None, converts dict into list of pairs, sorts, list -> tuple
        xb = tuple(sorted((x or {}).items()))
        yb = tuple(sorted((y or {}).items()))
        return cls(xb, yb)

    def render(self) -> np.ndarray:
        """Return this cube as a mask.

        Returns:
            A GRID_SIZE x GRID_SIZE boolean array indexed [y][x], True wherever
            every fixed bit matches. With no bits fixed, all True.
        """
        vals = np.arange(GRID_SIZE)

        in_x = np.ones(GRID_SIZE, dtype=bool)
        for bit, v in self.x_bits:
            in_x &= (vals >> bit) & 1 == v  # right shifts every number bit index, checks if last bit is equal to value

        in_y = np.ones(GRID_SIZE, dtype=bool)
        for bit, v in self.y_bits:
            in_y &= (vals >> bit) & 1 == v

        return in_x[None, :] & in_y[:, None]  # combines x row and y column

@dataclass(frozen=True)
class Parity:
    x_bits: tuple[int, ...]
    y_bits: tuple[int, ...]

    @classmethod
    def fix(cls, x=None, y=None) -> "Parity":
        # works if x or y are None, sorts, converts to tuple
        xb = tuple(sorted(x or {}))
        yb = tuple(sorted(y or {}))
        return cls(xb, yb)

    def render(self) -> np.ndarray:
        vals = np.arange(GRID_SIZE)

        in_x = np.zeros(GRID_SIZE, dtype=bool)
        for bit in self.x_bits:
            in_x ^= (vals >> bit) & 1 == 1 # shift and XORs 1 bits

        in_y = np.zeros(GRID_SIZE, dtype=bool)
        for bit in self.y_bits:
            in_y ^= (vals >> bit) & 1 == 1

        return in_x[None, :] ^ in_y[:, None]  # combines x row and y column

@dataclass(frozen=True)
class HalfPlane:
    a: int
    b: int
    c: int

    def render(self) -> np.ndarray:
        vals = np.arange(GRID_SIZE)

        in_x = vals[None, :]
        in_y = vals[:, None]
        return self.a*in_x + self.b*in_y <= self.c

@dataclass(frozen=True)
class And:
    children: tuple[Term, ...]

    def render(self) -> np.ndarray:
        mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        for t in self.children:
            mask &= t.render()
        return mask

@dataclass(frozen=True)
class Or:
    children: tuple[Term, ...]

    def render(self) -> np.ndarray:
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for t in self.children:
            mask |= t.render()
        return mask

@dataclass(frozen=True)
class Xor:
    children: tuple[Term, ...]

    def render(self) -> np.ndarray:
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for t in self.children:
            mask ^= t.render()
        return mask

@dataclass(frozen=True)
class Not:
    child: Term

    def render(self) -> np.ndarray:
        return ~self.child.render()

def cover_mask(terms) -> np.ndarray:
    """XOR every term's mask together to get the design's phase pattern.

    XOR, not OR. A pixel comes out True when an odd number of terms cover it,
    so stamping a region twice cancels it. That is what makes overlapping
    covers a tool instead of a bug, and it mirrors the circuit exactly: two
    phase flips on the same state multiply back to +1.

    Args:
        terms: Any iterable of objects with a ``render()`` method.

    Returns:
        A GRID_SIZE x GRID_SIZE boolean array indexed [y][x]. An empty design
        gives an all-False mask, since nothing has been stamped.
    """
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for t in terms:
        mask ^= t.render()
    return mask


def matches_target(terms) -> bool:
    """Return True if this design produces exactly the target image.

    This is the entire correctness check. The oracle is diagonal -- all it can
    do is attach a sign to each of the 4096 basis states -- so comparing masks
    is not an approximation of simulating the circuit, it *is* the result.
    Microseconds, and exact.

    It says nothing about cost. Depth, width and ancilla count depend on how
    the design is compiled, and are measured elsewhere.

    Args:
        terms: Any iterable of objects with a ``render()`` method.

    Returns:
        True if the XOR of all rendered terms equals TARGET.
    """
    return np.array_equal(cover_mask(terms), TARGET)

TERMS = {cls.__name__: cls for cls in (Rect, Cube, Parity, HalfPlane, And, Or, Xor, Not)}

def to_data(obj):
    """recurses through terms to keep dataclass names"""
    if is_dataclass(obj):
        return [type(obj).__name__, {f.name: to_data(getattr(obj, f.name)) for f in fields(obj)}]
    if isinstance(obj, (list, tuple)):
        return [to_data(o) for o in obj]
    return obj

def from_data(obj):
    """recurses through recovered json data and rebuilds terms"""
    if isinstance(obj, list):
        if len(obj) == 2 and isinstance(obj[0], str) and isinstance(obj[1], dict):
            cls = TERMS[obj[0]]
            return cls(**{k: from_data(v) for k,v in obj[1].items()})

        return tuple([from_data(o) for o in obj])

    return obj

def serialize(terms) -> str:
    """Returns a cannonical string for a given list of terms"""
    return json.dumps(to_data(terms), sort_keys=True, separators=(',', ':'))

def deserialize(data_str):
    data = json.loads(data_str)
    return from_data(data)

def fingerprint(terms) -> str:
    """Returns a hash string from a given list of terms"""
    return hashlib.sha256(serialize(terms).encode()).hexdigest()