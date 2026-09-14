from .mask import GRID_SIZE, COORD_BITS
from .terms import Rect, Cube, Parity, HalfPlane, And, Or, Xor, Not

PROXIES = {
    "P1": lambda f: f["n_terms"],
    "P2": lambda f: sum(f["controls"]),
    "P3": lambda f: sum(f["aligned_bits"]),
    "P4": lambda f: sum(22 + 25 * e + 2 * b + 137 * a for e, b, a in zip(f["aligned_bits"], f["bit_tests"], f["n_adders"])),
    "P5": lambda f: sum(m * (22 + 25 * e + 2 * b + 137 * a) for e, b, a, m in zip(f["aligned_bits"], f["bit_tests"], f["n_adders"], f["multiplier"])),
}

def features(terms):
    f = {}
    f["n_terms"] = len(terms)
    f["controls"] = tuple(_controls(t) for t in terms)
    f["n_adders"] = tuple(_n_adders(t) for t in terms)
    f["aligned_bits"] = tuple(_examine(t) for t in terms)
    f["bit_tests"] = tuple(_bit_tests(t) for t in terms)
    f["multiplier"] = tuple(_multiplier(t) for t in terms)
    return f


def _kids(term):
    """The children of any term. One place that knows `children` from `child`."""
    if isinstance(term, (And, Or, Xor)):
        return term.children
    if isinstance(term, Not):
        return (term.child,)
    return ()


def _walk(term):
    """Yield a term and every term nested inside it, at any depth."""
    yield term
    for kid in _kids(term):
        yield from _walk(kid)


def _node_reason(term, depth) -> "str|None":
    """Rules that are true of ONE term, looking only at that term.

    `depth` is 1 for a top-level term, 2 for its children, and so on. Only the
    Parity rule needs it -- a standalone Parity is free at any bit count
    (12 bits measured at depth 1, zero CX), while a nested one is refused at
    two bits. Same term, opposite answers.

    There is deliberately no rule about nesting depth itself. An earlier
    version refused everything at depth 3 on the strength of one measurement,
    and the gap sweep then found three depth-3 designs that synthesize fine:
    Not(And(R,R2)) at 889, Xor(R3, And(R,R2)) at 1723, Xor(Not(R),R2) at 1897.
    What actually refuses is an `And` with a container child, which is the
    rule below. `Xor` and `Not` carry containers without complaint.
    """
    if isinstance(term, Or):
        return "Or (refused even with the two cheapest possible children)"

    if isinstance(term, (And, Xor)) and len(term.children) >= 3:
        return f"{type(term).__name__} with {len(term.children)} children (max 2)"

    if isinstance(term, And) and any(isinstance(k, (And, Or, Xor, Not)) for k in term.children):
        return "And with a container child (Xor and Not allow them; And does not)"

    if isinstance(term, Cube) and (len(term.x_bits) >= 3 or len(term.y_bits) >= 3):
        # A Cube's predicate ANDs the x bits into one value and the y bits
        # into another, then ANDs those. Three bits on a single axis is a
        # three-deep chain and wants 20 qubits; two on x plus one on y is the
        # same three bits split into two shallow chains, and costs 14.
        return "3+ Cube bits on one axis (max 2 per axis)"

    if isinstance(term, HalfPlane) and (abs(term.a) > 1 or abs(term.b) > 1):
        # a*x with |a| > 1 is a constant multiplier, not just an adder.
        # HalfPlane(3, 1, 60) wants 21 qubits; every coefficient in {0, +-1}
        # measured so far fits.
        return f"HalfPlane coefficient magnitude > 1 (a={term.a}, b={term.b})"

    if isinstance(term, Parity) and depth > 1:
        bits = len(term.x_bits) + len(term.y_bits)
        if bits >= 2:
            return f"nested Parity with {bits} bits (max 1 when nested, unlimited at top level)"

    return None


def _aggregate_reason(term) -> "str|None":
    """Rules true of a WHOLE top-level term, which no single node can see.

    And(Cube2, Cube2) is refused even though neither Cube has four bits --
    the *term* has four. The budget is shared across one top-level expression,
    because everything inside one `control` is evaluated in parallel and every
    sub-expression holds its own scratch at the same moment.

    It is per top-level term, not per design: 32 separate Rects synthesize
    fine, because sequential terms compute, use and release in turn.
    """
    nodes = list(_walk(term))
    cube_bits = sum(len(n.x_bits) + len(n.y_bits) for n in nodes if isinstance(n, Cube))
    rects = sum(1 for n in nodes if isinstance(n, Rect))
    adders = sum(1 for n in nodes
                 if isinstance(n, HalfPlane) and n.a != 0 and n.b != 0)

    if cube_bits >= 4:
        return f"{cube_bits} Cube bits in one term (max 3 total, and max 2 per axis)"
    if rects and cube_bits >= 2:
        return "a Rect together with a Cube of 2+ bits"
    if adders >= 2:
        return f"{adders} adder half-planes in one term (max 1)"
    return None


def refusal_reason(terms) -> "str|None":
    """Why Classiq would refuse this design, or None if it should synthesize.

    A lookup of measured limits, not a width model. Every rule traces to a
    specific refusal in logs/experiments.jsonl; see experiments/sweep.py, the
    `limits_*` groups. With this few data points a model would confidently
    reject combinations nobody has tested.

    Deliberately permissive where the evidence runs out. A wrong "fits" costs
    one 40-second cloud call and tells you so. A wrong "does not fit" deletes
    a design from the search space and never tells you what it would have
    scored -- the error you cannot see is the expensive one.

    Args:
        terms: a design -- the same list of terms the proxies take.

    Returns:
        A short reason naming the offending term, or None if the design fits.
    """
    for i, term in enumerate(terms):
        reason = _aggregate_reason(term) or _node_reason_deep(term)
        if reason:
            return f"term {i} ({type(term).__name__}): {reason}"
    return None


def _node_reason_deep(term, depth=1) -> "str|None":
    """First node rule broken anywhere in this term, depth-first."""
    reason = _node_reason(term, depth)
    if reason:
        return reason
    for kid in _kids(term):
        reason = _node_reason_deep(kid, depth + 1)
        if reason:
            return reason
    return None


def fits(terms) -> bool:
    """True when Classiq should synthesize this design within 18 qubits."""
    return refusal_reason(terms) is None


def _controls(term, nested=False) -> int:
    """returns the total number of controls that act on a term"""
    if isinstance(term, Rect):
        return (term.min_x > 0) + (term.max_x < GRID_SIZE - 1) + (term.min_y > 0) + (term.max_y < GRID_SIZE - 1)
    if isinstance(term, Cube):
        return len(term.x_bits) + len(term.y_bits)
    if isinstance(term, Parity):
        if not nested:
            return 0
        return len(term.x_bits) + len(term.y_bits)
    if isinstance(term, HalfPlane):
        return 1
    if isinstance(term, And):
        sum = len(term.children)
        for t in term.children:
            sum += _controls(t, nested=True)
        return sum
    if isinstance(term, Or):
        sum = len(term.children)
        for t in term.children:
            sum += _controls(t, nested=True)
        return sum
    if isinstance(term, Xor):
        sum = len(term.children)
        for t in term.children:
            sum += _controls(t, nested=True)
        return sum
    if isinstance(term, Not):
        return 1 + _controls(term.child, nested=True)
    
    raise TypeError(f"no control count for {type(term).__name__}")

def _n_adders(term) -> int:
    if isinstance(term, HalfPlane):
        if term.a != 0 and term.b !=0:
            return 1
    if isinstance(term, And):
        sum = 0
        for t in term.children:
            sum += _n_adders(t)
        return sum
    if isinstance(term, Or):
        sum = 0
        for t in term.children:
            sum += _n_adders(t)
        return sum
    if isinstance(term, Xor):
        sum = 0
        for t in term.children:
            sum += _n_adders(t)
        return sum
    if isinstance(term, Not):
        return _n_adders(term.child)

    return 0

def _examine_bits(value, upper):
    bin_str = format(value, "06b")
    count = 0
    if upper:
        for c in reversed(bin_str):
            if c == '1':
                count += 1
            else: break
    else: 
        for c in reversed(bin_str):
            if c == '0':
                count += 1
            else: break
    return COORD_BITS - count

def _examine(term) -> int:
    if isinstance(term, Rect):
        return _examine_bits(term.min_x, False) + _examine_bits(term.max_x, True) + _examine_bits(term.min_y, False) + _examine_bits(term.max_y, True)
    if isinstance(term, HalfPlane):
        # not well defined for halfplane
        return max(0, min(_examine_bits(term.c, True), COORD_BITS))
    if isinstance(term, And):
        sum = 0
        for t in term.children:
            sum += _examine(t)
        return sum
    if isinstance(term, Or):
        sum = 0
        for t in term.children:
            sum += _examine(t)
        return sum
    if isinstance(term, Xor):
        sum = 0
        for t in term.children:
            sum += _examine(t)
        return sum
    if isinstance(term, Not):
        return _examine(term.child)

    return 0

def _bit_tests(term, nested=False):
    if isinstance(term, Cube):
        return len(term.x_bits) + len(term.y_bits)
    if isinstance(term, Parity):
        if nested:
            return len(term.x_bits) + len(term.y_bits)
        return 0
    if isinstance(term, And):
        sum = 0
        for t in term.children:
            sum += _bit_tests(t, nested=True)
        return sum
    if isinstance(term, Or):
        sum = 0
        for t in term.children:
            sum += _bit_tests(t, nested=True)
        return sum
    if isinstance(term, Xor):
        sum = 0
        for t in term.children:
            sum += _bit_tests(t, nested=True)
        return sum
    if isinstance(term, Not):
        return _bit_tests(term.child, nested=True)

    return 0

def _multiplier(term):
    if isinstance(term, And):
        mult = 1
        for t in term.children:
            mult *= _multiplier(t)
        return mult
    if isinstance(term, Or):
        mult = 2.4
        for t in term.children:
            mult *= _multiplier(t)
        return mult
    if isinstance(term, Xor):
        mult = 2.4
        for t in term.children:
            mult *= _multiplier(t)
        return mult
    if isinstance(term, Not):
        return 2.5*_multiplier(term.child)
    
    return 1