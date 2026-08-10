from __future__ import annotations

from typing import Mapping

from federated.types.fl_types import StateDict

def assert_same_parameter_keys(
    reference: StateDict,
    candidate: StateDict,
    *,
    who: str = "",
    max_show: int = 8,
) -> None:
    """
    Check if client contains all parameters
    """
    ref_keys, cand_keys = set(reference.keys()), set(candidate.keys())
    if ref_keys != cand_keys:
        missing = sorted(ref_keys - cand_keys)
        extra = sorted(cand_keys - ref_keys)
        raise KeyError(f"{who}: layer parameters mismatch | missing({len(missing)}): {missing[:max_show]} | extra({len(extra)}): {extra[:max_show]}")