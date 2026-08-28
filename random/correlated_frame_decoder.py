"""Check whether joint CSS syndromes distinguish frame-correlated errors.

The input error model is the one used in the accompanying note.  Before a
transversal phase gate, an error is written as X(a) Z(b), with the two
component weights bounded separately.  Ignoring phase, transversal S or
S-dagger maps it to Y(a) Z(b) = X(a) Z(a xor b).

Two notions of ambiguity are reported:

* logical ambiguity: the same observed record is compatible with two data
  errors whose difference is a non-trivial logical Pauli.  No recovery can
  correct the corresponding conditional error set.
* exact-reset ambiguity: the same observed record is compatible with two
  data errors that differ by more than a stabilizer.  A single Pauli selected
  from that record cannot return every candidate exactly to the codespace.

With perfect syndrome bits these notions coincide.  With readout-bit errors,
an exact-reset ambiguity need not be a logical ambiguity: the two candidates
may retain different correctable syndromes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from spiderstate.utils import load_qecc


@dataclass(frozen=True)
class ErrorCandidate:
    """One data-and-readout event considered by the checker."""

    pre_x: int
    pre_z: int
    measurement_error: int
    post_pauli: int
    true_syndrome: int
    observed_record: int
    order: int


@dataclass(frozen=True)
class AmbiguityWitness:
    """Two candidates that cannot be identified in the requested sense."""

    first: ErrorCandidate
    second: ErrorCandidate


@dataclass(frozen=True)
class CorrelatedDecoderReport:
    code: str
    n: int
    k: int
    d: int
    max_order: int
    syndrome_bits: int
    includes_measurement_errors: bool
    transversal_s_preserves_code: bool
    candidate_count: int
    observed_record_count: int
    logical_ambiguity_records: int
    exact_reset_ambiguity_records: int
    logical_witnesses: tuple[AmbiguityWitness, ...]
    exact_reset_witnesses: tuple[AmbiguityWitness, ...]

    @property
    def logically_decodable(self) -> bool:
        return self.logical_ambiguity_records == 0

    @property
    def exactly_resettable(self) -> bool:
        return self.exact_reset_ambiguity_records == 0


def _rows_to_masks(matrix: np.ndarray) -> list[int]:
    return [
        sum(int(bit) << column for column, bit in enumerate(row))
        for row in np.asarray(matrix, dtype=np.uint8)
    ]


def _linear_basis(vectors: Iterable[int]) -> dict[int, int]:
    """Return a highest-pivot GF(2) basis represented by Python integers."""

    basis: dict[int, int] = {}
    for vector in vectors:
        value = int(vector)
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return basis


def _coset_remainder(vector: int, basis: dict[int, int]) -> int:
    """Canonical remainder modulo the span of ``basis``."""

    value = int(vector)
    for pivot in sorted(basis, reverse=True):
        if (value >> pivot) & 1:
            value ^= basis[pivot]
    return value


def _syndrome(x_mask: int, z_mask: int, h_x: list[int], h_z: list[int]) -> int:
    """Pack (Z-check outcomes, X-check outcomes) into one integer."""

    result = 0
    offset = 0
    for check in h_z:
        result |= ((x_mask & check).bit_count() & 1) << offset
        offset += 1
    for check in h_x:
        result |= ((z_mask & check).bit_count() & 1) << offset
        offset += 1
    return result


def _support_masks_by_weight(length: int, max_weight: int) -> list[list[int]]:
    masks: list[list[int]] = []
    for weight in range(max_weight + 1):
        masks.append(
            [sum(1 << qubit for qubit in support) for support in combinations(range(length), weight)]
        )
    return masks


def _transversal_s_preserves_code(
    h_x_matrix: np.ndarray,
    h_x: list[int],
    h_z: list[int],
    n: int,
) -> bool:
    """Check the Pauli image and stabilizer signs for uniform transversal S."""

    stabilizer_basis = _linear_basis(h_x + [row << n for row in h_z])
    for row_array, row_mask in zip(h_x_matrix, h_x, strict=True):
        # S X(h) S^dagger = i^|h| X(h) Z(h).  A +1 X stabilizer
        # must map to a +1 stabilizer, so |h| must be 0 modulo 4.
        if int(np.sum(row_array)) % 4 != 0:
            return False
        image = row_mask | (row_mask << n)
        if _coset_remainder(image, stabilizer_basis) != 0:
            return False
    return True


def _first_distinct_coset_pair(
    candidates_by_coset: dict[int, ErrorCandidate],
) -> AmbiguityWitness | None:
    values = list(candidates_by_coset.values())
    if len(values) < 2:
        return None
    return AmbiguityWitness(values[0], values[1])


def analyze_correlated_frame_decoder(
    code: str,
    *,
    method: str | None = "FAO",
    max_order: int | None = None,
    include_measurement_errors: bool = False,
    max_witnesses: int = 3,
) -> CorrelatedDecoderReport:
    """Analyze joint decoding after transversal S for a Spiderstate CSS code.

    A pre-S data event X(a)Z(b) is assigned component order
    ``max(weight(a), weight(b))``.  When readout errors are enabled, each
    flipped syndrome bit costs one additional fault, so retained events obey

        max(weight(a), weight(b)) + weight(measurement_error) <= max_order.

    This is deliberately a code-capacity/readout model.  Correlated hook
    errors from a concrete extraction circuit must be supplied by a separate
    circuit-level enumerator.
    """

    _, h_x_matrix, h_z_matrix, l_x_matrix, _, distance = load_qecc(code, method)
    h_x_matrix = np.asarray(h_x_matrix, dtype=np.uint8)
    h_z_matrix = np.asarray(h_z_matrix, dtype=np.uint8)
    l_x_matrix = np.asarray(l_x_matrix, dtype=np.uint8)

    if h_x_matrix.ndim != 2 or h_z_matrix.ndim != 2:
        raise ValueError("H_x and H_z must be two-dimensional matrices")
    if h_x_matrix.shape[1] != h_z_matrix.shape[1]:
        raise ValueError("H_x and H_z must have the same block length")
    if np.any((h_x_matrix @ h_z_matrix.T) % 2):
        raise ValueError(f"{code} is not CSS: H_x H_z^T is nonzero")

    n = int(h_x_matrix.shape[1])
    k = int(l_x_matrix.shape[0])
    t = (int(distance) - 1) // 2 if max_order is None else int(max_order)
    if t < 0:
        raise ValueError("max_order must be non-negative")

    h_x = _rows_to_masks(h_x_matrix)
    h_z = _rows_to_masks(h_z_matrix)
    syndrome_bits = len(h_z) + len(h_x)
    stabilizer_basis = _linear_basis(h_x + [row << n for row in h_z])
    data_supports = _support_masks_by_weight(n, t)
    measurement_supports = _support_masks_by_weight(syndrome_bits, t)

    # observed -> true syndrome -> stabilizer coset -> representative event
    records: dict[int, dict[int, dict[int, ErrorCandidate]]] = {}
    candidate_count = 0

    for x_weight in range(t + 1):
        for z_weight in range(t + 1):
            data_order = max(x_weight, z_weight)
            if data_order > t:
                continue
            remaining_measurement_order = t - data_order if include_measurement_errors else 0
            measurement_patterns = (
                pattern
                for weight in range(remaining_measurement_order + 1)
                for pattern in measurement_supports[weight]
            )
            # Materialise because the same patterns are reused for every data pair.
            measurement_patterns = tuple(measurement_patterns)

            for pre_x in data_supports[x_weight]:
                for pre_z in data_supports[z_weight]:
                    # S and S-dagger have the same Pauli action modulo phase.
                    post_x = pre_x
                    post_z = pre_x ^ pre_z
                    post_pauli = post_x | (post_z << n)
                    true_syndrome = _syndrome(post_x, post_z, h_x, h_z)
                    coset = _coset_remainder(post_pauli, stabilizer_basis)

                    for measurement_error in measurement_patterns:
                        order = data_order + measurement_error.bit_count()
                        observed = true_syndrome ^ measurement_error
                        candidate = ErrorCandidate(
                            pre_x=pre_x,
                            pre_z=pre_z,
                            measurement_error=measurement_error,
                            post_pauli=post_pauli,
                            true_syndrome=true_syndrome,
                            observed_record=observed,
                            order=order,
                        )
                        candidate_count += 1
                        records.setdefault(observed, {}).setdefault(true_syndrome, {}).setdefault(
                            coset, candidate
                        )

    logical_ambiguity_records = 0
    exact_reset_ambiguity_records = 0
    logical_witnesses: list[AmbiguityWitness] = []
    exact_reset_witnesses: list[AmbiguityWitness] = []

    for true_syndrome_buckets in records.values():
        logical_witness: AmbiguityWitness | None = None
        all_cosets: dict[int, ErrorCandidate] = {}

        for candidates_by_coset in true_syndrome_buckets.values():
            all_cosets.update(candidates_by_coset)
            if logical_witness is None:
                logical_witness = _first_distinct_coset_pair(candidates_by_coset)

        if logical_witness is not None:
            logical_ambiguity_records += 1
            if len(logical_witnesses) < max_witnesses:
                logical_witnesses.append(logical_witness)

        exact_witness = _first_distinct_coset_pair(all_cosets)
        if exact_witness is not None:
            exact_reset_ambiguity_records += 1
            if len(exact_reset_witnesses) < max_witnesses:
                exact_reset_witnesses.append(exact_witness)

    return CorrelatedDecoderReport(
        code=code,
        n=n,
        k=k,
        d=int(distance),
        max_order=t,
        syndrome_bits=syndrome_bits,
        includes_measurement_errors=include_measurement_errors,
        transversal_s_preserves_code=_transversal_s_preserves_code(h_x_matrix, h_x, h_z, n),
        candidate_count=candidate_count,
        observed_record_count=len(records),
        logical_ambiguity_records=logical_ambiguity_records,
        exact_reset_ambiguity_records=exact_reset_ambiguity_records,
        logical_witnesses=tuple(logical_witnesses),
        exact_reset_witnesses=tuple(exact_reset_witnesses),
    )


def _mask_support(mask: int, length: int) -> list[int]:
    return [qubit + 1 for qubit in range(length) if (mask >> qubit) & 1]


def _format_candidate(candidate: ErrorCandidate, n: int, syndrome_bits: int) -> str:
    post_x = candidate.post_pauli & ((1 << n) - 1)
    post_z = candidate.post_pauli >> n
    return (
        f"pre X={_mask_support(candidate.pre_x, n)}, "
        f"pre Z={_mask_support(candidate.pre_z, n)}, "
        f"post X={_mask_support(post_x, n)}, "
        f"post Z={_mask_support(post_z, n)}, "
        f"meas={_mask_support(candidate.measurement_error, syndrome_bits)}, "
        f"order={candidate.order}"
    )


def _print_report(report: CorrelatedDecoderReport) -> None:
    mode = "data + syndrome-bit errors" if report.includes_measurement_errors else "perfect syndromes"
    print(f"{report.code} [[{report.n},{report.k},{report.d}]], order <= {report.max_order}, {mode}")
    print(f"  transversal S preserves code: {report.transversal_s_preserves_code}")
    print(f"  candidates / records: {report.candidate_count} / {report.observed_record_count}")
    print(f"  logical-ambiguity records: {report.logical_ambiguity_records}")
    print(f"  exact-reset ambiguity records: {report.exact_reset_ambiguity_records}")
    for label, witnesses in (
        ("logical", report.logical_witnesses),
        ("exact-reset", report.exact_reset_witnesses),
    ):
        for index, witness in enumerate(witnesses, start=1):
            print(f"  {label} witness {index}:")
            print(f"    A: {_format_candidate(witness.first, report.n, report.syndrome_bits)}")
            print(f"    B: {_format_candidate(witness.second, report.n, report.syndrome_bits)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codes", nargs="+", help="Spiderstate code names, e.g. 7_1_3 17_1_5")
    parser.add_argument("--method", default="FAO", help="QECC library passed to load_qecc")
    parser.add_argument("--max-order", type=int, default=None, help="defaults to floor((d-1)/2)")
    parser.add_argument(
        "--measurement-errors",
        action="store_true",
        help="include independent syndrome-bit flips in the total fault budget",
    )
    args = parser.parse_args()

    for index, code in enumerate(args.codes):
        if index:
            print()
        _print_report(
            analyze_correlated_frame_decoder(
                code,
                method=args.method,
                max_order=args.max_order,
                include_measurement_errors=args.measurement_errors,
            )
        )


if __name__ == "__main__":
    main()
