import classiq
from classiq import *
from classiq.qmod.symbolic import pi

from .mask import COORD_BITS
from .terms import Rect, Cube, Parity, HalfPlane, And, Or, Xor, Not

COMPILER_VERSION = 1

#classiq.authenticate()

def predicate(term, x: Const[QNum], y: Const[QNum]):
    """Given a term, returns a Qmod boolean expression, condition for where to apply shift"""
    if isinstance(term, Rect):
        return ((y >= term.min_y) & (y <= term.max_y)) & ((x >= term.min_x) & (x <= term.max_x))

    if isinstance(term, Cube):
        x_con = True
        y_con = True
        for bit, v in term.x_bits:
            x_con &= (x >> bit) & 1 == v
        for bit, v in term.y_bits:
            y_con &= (y >> bit) & 1 == v

        return x_con & y_con

    if isinstance(term, Parity):
        x_con = False
        y_con = False
        for bit in term.x_bits:
            x_con ^= (x >> bit) & 1 == 1
        for bit in term.y_bits:
            y_con ^= (y >> bit) & 1 == 1

        return x_con ^ y_con

    if isinstance(term, HalfPlane):
        return term.a*x + term.b*y <= term.c

    if isinstance(term, And):
        p = True
        for t in term.children:
            p &= predicate(t, x, y)
        return p

    if isinstance(term, Or):
        p = False
        for t in term.children:
            p |= predicate(t, x, y)
        return p

    if isinstance(term, Xor):
        p = False
        for t in term.children:
            p ^= predicate(t, x, y)
        return p

    if isinstance(term, Not):
        return predicate(term.child, x, y) == 0
                

    # unimplemented term
    raise TypeError(f"no predicate rule for {type(term).__name__}")

def emit(term, x: Const[QNum], y: Const[QNum], i) -> None:
    """Applies phase shift to predicate"""
    if isinstance(term, Parity):
        xb = QArray(f"xb{i}") # crashes if there are multiple instances of a QArray with same name, need diffbut
        bind(x, xb) # binds x the QNum -> quantum registry of 6 qubits
        for bit in term.x_bits:
            Z(xb[bit])
        bind (xb, x) # get back x
        yb = QArray(f"yb{i}")
        bind(y, yb)
        for bit in term.y_bits:
            Z(yb[bit])
        bind (yb, y)
        return
        
    control(predicate(term, x, y), lambda: phase(pi))

def build_main(terms):
    """Return a Qmod main that applies input terms as a phase oracle."""
    # from notebook
    @qperm
    def oracle(x: Const[QNum], y: Const[QNum]) -> None:
        # emits for all terms, XOR of all terms
        for i, t in enumerate(terms):
            emit(t, x, y, i)

    @qfunc
    def main(x: Output[QNum[COORD_BITS]], y: Output[QNum[COORD_BITS]],) -> None:
        """Prepare a full-support synthesis context and apply the oracle."""
        allocate(x)
        allocate(y)
        hadamard_transform(x)
        hadamard_transform(y)
        oracle(x, y)

    # return main function
    return main
