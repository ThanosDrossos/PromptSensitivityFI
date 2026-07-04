"""FI_spec — normative question specificity in bits (AmbigQA pivot).

Dataset-side, model-free: how many bits the QUESTION TEXT removes from the
answer space, relative to the raw question's full ambiguity. With m0 valid
interpretations of the ambiguous question and m_valid still admitted by the
(possibly disambiguated) text:

    FI_spec = log2(m0 / m_valid)

level 0 (ambiguous Q):      m_valid = m0 -> 0.0 bits
level 1 (disambiguated Q_i): m_valid = 1  -> log2(m0) bits

Deliberately NOT computed in the orchestrator: the orchestrator owns the
model-side math; FI_spec is a property of the dataset row and is passed through
by the driver (scripts/run_specificity.py).
"""

from __future__ import annotations

import math


def fi_spec_bits(m0: int, m_valid: int) -> float:
    """Bits the question removes from the answer space (0.0 on degenerate input)."""
    if m0 <= 0 or m_valid <= 0:
        return 0.0
    return math.log2(m0 / m_valid)
