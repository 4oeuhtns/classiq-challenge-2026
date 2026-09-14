from dataclasses import dataclass

import numpy as np
import re
import warnings
import classiq
from classiq import *

from .mask import COORD_BITS, GRID_SIZE
from .terms import cover_mask

# Numerical precision thresholds for statevector verification.
STATE_ERROR_THRESHOLD = 1e-10
NORMALIZATION_ERROR_THRESHOLD = 1e-10

@dataclass(frozen=True)
class Verification:
    """verification dataclass to return verification results"""
    seed: int
    width: int
    depth: int
    cx_count: int
    max_error: float
    ancilla_error: float
    normalization_error: float

    @property
    def ok(self) -> bool:
        return max(self.max_error, self.ancilla_error) < STATE_ERROR_THRESHOLD
    


def qasm_metrics(source: str) -> tuple[int, int, int]:
    """Return width, depth, and CX count for a u3/cx QASM 2.0 file."""
    statements = [
        statement.strip()
        for statement in re.sub(r"//[^\n]*", "", source).split(";")
        if statement.strip()
    ]
    if statements[:2] != ["OPENQASM 2.0", 'include "qelib1.inc"']:
        raise ValueError("Expected an OpenQASM 2.0 file using qelib1.inc")

    qreg_match = (
        re.fullmatch(r"qreg\s+q\s*\[\s*(\d+)\s*\]", statements[2])
        if len(statements) >= 3
        else None
    )
    if qreg_match is None:
        raise ValueError("Expected exactly one register named q")

    width = int(qreg_match.group(1))
    if not 12 <= width <= 18:
        raise ValueError(f"QASM width must be between 12 and 18, got {width}")
    qubit_depths = [0] * width
    cx_count = 0

    for statement in statements[3:]:
        u3_match = re.fullmatch(
            r"u3\s*\([^,]+,[^,]+,[^,]+\)\s+q\s*\[\s*(\d+)\s*\]",
            statement,
        )
        cx_match = re.fullmatch(
            r"cx\s+q\s*\[\s*(\d+)\s*\]\s*,"
            r"\s*q\s*\[\s*(\d+)\s*\]",
            statement,
        )
        if u3_match:
            qubits = (int(u3_match.group(1)),)
        elif cx_match:
            qubits = tuple(map(int, cx_match.groups()))
            cx_count += 1
        else:
            raise ValueError(f"Unsupported QASM statement: {statement}")

        if any(qubit >= width for qubit in qubits):
            raise ValueError(
                f"Qubit index outside q[0]...q[{width - 1}]: {statement}"
            )
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Gate operands must be distinct: {statement}")

        layer = max(qubit_depths[qubit] for qubit in qubits) + 1
        for qubit in qubits:
            qubit_depths[qubit] = layer

    return width, max(qubit_depths, default=0), cx_count


def dense_classiq_statevector(frame, width: int) -> tuple[np.ndarray, float]:
    """Validate Classiq output and reconstruct the full physical vector."""
    statevector = np.zeros(1 << width, dtype=np.complex128)
    seen_basis_states: set[int] = set()

    for row in frame.itertuples(index=False):
        bitstring = str(row.bitstring).replace(" ", "")
        if len(bitstring) != width or set(bitstring) - {"0", "1"}:
            raise ValueError(f"Unexpected Classiq bitstring: {row.bitstring}")
        basis_index = int(bitstring, 2)
        if basis_index in seen_basis_states:
            raise ValueError(f"Duplicate Classiq basis state: {bitstring}")

        amplitude = complex(row.amplitude)
        if not np.isfinite(amplitude.real) or not np.isfinite(amplitude.imag):
            raise ValueError(f"Non-finite Classiq amplitude at {bitstring}")
        seen_basis_states.add(basis_index)
        statevector[basis_index] = amplitude

    if not seen_basis_states:
        raise ValueError("Classiq simulator returned an empty statevector")

    normalization_error = abs(np.vdot(statevector, statevector).real - 1)
    if normalization_error >= NORMALIZATION_ERROR_THRESHOLD:
        raise ValueError(
            f"Classiq statevector is not normalized: {normalization_error:.3e}"
        )
    return statevector, normalization_error

def verify(terms, qasm_str, seed: int):
    """checks QASM submission against the cover mask made by the list of terms"""
    submission_width, depth, cx_count = qasm_metrics(qasm_str)

    logical_size = GRID_SIZE**2
    rng = np.random.default_rng(seed=seed)
    random_phases = rng.uniform(
        -np.pi, np.pi, size=(3, 2 * COORD_BITS)
    )
    target_phases = ((-2*cover_mask(terms))+1).astype(np.complex128).ravel() # converts from (0, 1) to (1, -1), convert to complex
    source_without_comments = re.sub(
        r"//[^\n]*",
        lambda match: " " * len(match.group()),
        qasm_str,
    )
    qreg_matches = list(
        re.finditer(
            rf"\bqreg\s+q\s*\[\s*{submission_width}\s*\]\s*;",
            source_without_comments,
        )
    )
    if len(qreg_matches) != 1:
        raise ValueError("QASM must contain exactly one matching q register")
    insertion_index = qreg_matches[0].end()
    basis = np.arange(logical_size, dtype=np.uint16)
    max_error = ancilla_error = normalization_error = 0.0
    shared_global_phase = 1 + 0j

    # Raw QASM has no named Classiq outputs, so Classiq labels its wires as
    # auxiliary. We validate the actual ancilla wires explicitly below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Non-zero amplitudes were detected in states with non-zero auxiliary values.*",
        )
        for test_index, phases in enumerate(random_phases):
            preparation = [
                f"u3(pi/2,{phase:.17g},0) q[{qubit}];"
                for qubit, phase in enumerate(phases)
            ]
            test_qasm = (
                qasm_str[:insertion_index]
                + "\n"
                + "\n".join(preparation)
                + qasm_str[insertion_index:]
            )

            with ExecutionSession(
                test_qasm,
                backend="classiq/simulator",
                transpilation_option=TranspilationOption.DECOMPOSE,
            ) as session:
                classiq_frame = session.calculate_state_vector(
                    amplitude_threshold=0.0
                )

            statevector, state_normalization_error = dense_classiq_statevector(
                classiq_frame, submission_width
            )
            normalization_error = max(
                normalization_error, state_normalization_error
            )

            # Ancillas are the higher-index wires, so indices below 4096 have
            # q[12:] = |0...0> under the required little-endian convention.
            ancilla_error = max(
                ancilla_error,
                np.max(np.abs(statevector[logical_size:]), initial=0.0),
            )

            basis_phases = np.zeros(logical_size)
            for qubit, phase in enumerate(phases):
                basis_phases += ((basis >> qubit) & 1) * phase

            expected_state = np.zeros(1 << submission_width, dtype=np.complex128)
            expected_state[:logical_size] = (
                np.exp(1j * basis_phases) / np.sqrt(logical_size) * target_phases
            )
            if test_index == 0:
                overlap = np.vdot(expected_state, statevector)
                if abs(overlap) <= 1e-12:
                    raise ValueError(
                        "Cannot determine a global phase from Classiq results"
                    )
                shared_global_phase = overlap / abs(overlap)

            max_error = max(
                max_error,
                np.max(np.abs(statevector - shared_global_phase * expected_state)),
            )

    return Verification(seed, submission_width, depth, cx_count, max_error, ancilla_error, normalization_error)