"""Dataset loaders + Pydantic schemas. See `Research_Design_v3` §2."""

from .schemas import HotpotParagraph, HotpotSupportingFact, MultiHopQuestion
from .load_hotpotqa import load_hotpotqa_validation, parse_hotpotqa_record
from .load_2wiki import load_twiki_validation, parse_twiki_record
from .sample_questions import stratified_sample

__all__ = [
    "HotpotParagraph",
    "HotpotSupportingFact",
    "MultiHopQuestion",
    "load_hotpotqa_validation",
    "parse_hotpotqa_record",
    "load_twiki_validation",
    "parse_twiki_record",
    "stratified_sample",
]
