"""FI_spec bit cases (pivot spec §5)."""

from __future__ import annotations

from prompt_sensitivity.metrics.fi_spec import fi_spec_bits


def test_ambiguous_level_is_zero_bits():
    assert fi_spec_bits(4, 4) == 0.0


def test_disambiguated_level_is_log2_m0():
    assert fi_spec_bits(4, 1) == 2.0


def test_unambiguous_question_is_zero_bits():
    assert fi_spec_bits(1, 1) == 0.0


def test_degenerate_inputs_floor_to_zero():
    assert fi_spec_bits(0, 1) == 0.0
    assert fi_spec_bits(4, 0) == 0.0
    assert fi_spec_bits(-1, -1) == 0.0
