"""Role-templated paraphrase prompts (Razavi 2025 ECIR PromptSET pattern).

PromptSET (arXiv:2502.06065) generates paraphrases by asking the model to
rewrite a query in a target *persona's* voice while preserving the answer set.
The role set below (4 design-doc-pinned + 4 added 2026-06-29 to widen surface
variety) all follow the same structure (persona + invariance constraint +
single-line output) per Razavi §3.1. Which roles a run actually uses is selected
by `config.paraphrases.templates`, so the set is trivially adjustable.

The system prompt is identical across roles; only the persona description and
a stylistic hint differ. Output is always one line, no preamble, no
explanations — that keeps post-processing trivial and avoids generator-side
"helpful" framing that would itself be a paraphrase artifact.
"""

from __future__ import annotations

from typing import Sequence

from ..models.schemas import ChatMessage
from .schemas import RoleName


ROLE_NAMES: Sequence[RoleName] = (
    "neutral", "journalist", "casual_user", "domain_expert",
    # 2026-06-29: four additional registers to widen surface-form variety. They
    # span the same principled axis (persona-conditioned rewrite preserving the
    # answer set); the NLI + gold-constraint filters still guarantee semantic
    # equivalence, so they add diversity WITHOUT answer-set bias. More personas =>
    # more distinct accepted paraphrases / fewer dropped questions, at the same
    # EVAL cost (the eval is capped by max_paraphrases, not by persona count;
    # only the one-off paraphrase-prep job does more generation).
    "student", "terse_keyword", "formal_academic", "second_language",
)


_PERSONA: dict[RoleName, str] = {
    "neutral": (
        "Rewrite the question in clear, neutral, encyclopedia-style English. "
        "Avoid colloquialisms, opinions, or stylistic flourish."
    ),
    "journalist": (
        "Rewrite the question as a news journalist would phrase it for a "
        "factual article. Be precise, attribute nothing, use a professional "
        "register, and prefer active voice."
    ),
    "casual_user": (
        "Rewrite the question as a casual user might type it into a chatbot "
        "or web search box. Informal but not slang-heavy; you may drop "
        "articles or use contractions; questions phrased as imperatives or "
        "even fragments are fine, as long as the meaning is preserved."
    ),
    "domain_expert": (
        "Rewrite the question as a domain expert would phrase it among "
        "colleagues. Precise terminology, slightly higher register, can "
        "assume some shared background but must remain self-contained."
    ),
    "student": (
        "Rewrite the question as an inquisitive student studying the topic "
        "would ask it: plain and direct, possibly opening with 'Can you tell "
        "me' or 'I want to know'. Keep every entity and constraint that fixes "
        "the answer."
    ),
    "terse_keyword": (
        "Rewrite the question as a terse keyword / search-style query: "
        "telegraphic, drop articles and filler words, but KEEP every named "
        "entity, relation, and constraint that determines the answer. It need "
        "not be a grammatical sentence."
    ),
    "formal_academic": (
        "Rewrite the question in a formal academic register, as in a scholarly "
        "reference work: complete sentences, precise wording, no contractions "
        "or colloquialisms."
    ),
    "second_language": (
        "Rewrite the question as a careful non-native English speaker would: "
        "simple, grammatical sentences and common vocabulary, phrasing that may "
        "be slightly literal, with the meaning fully preserved."
    ),
    # ------------------------------------------------------------------ #
    # R6 width dial (2026-08-07) — NARROW arm: minimal-edit rewrites.     #
    # The three instructions target DIFFERENT minimal edits so the arm    #
    # can still fill a 10-slot universe under the >=6 edit-distance dedup.#
    # ------------------------------------------------------------------ #
    "minimal_synonym": (
        "Rewrite the question changing AS LITTLE AS POSSIBLE: replace exactly "
        "one or two content words with close synonyms and keep every other "
        "word, the word order, and the punctuation identical. The result must "
        "differ from the original but only barely."
    ),
    "minimal_reorder": (
        "Rewrite the question changing AS LITTLE AS POSSIBLE: keep the same "
        "words wherever you can and only adjust the word order or the "
        "auxiliary construction (e.g. 'Where did they film X' <-> 'Where was "
        "X filmed'). Do not introduce new content words."
    ),
    "minimal_function_words": (
        "Rewrite the question changing AS LITTLE AS POSSIBLE: only adjust "
        "function words — contractions, articles, prepositions, or an "
        "opening auxiliary — and keep every content word identical."
    ),
    # ------------------------------------------------------------------ #
    # R6 width dial — WIDE arm: fluent register/syntax/indirectness       #
    # shifts. FLUENT ONLY by decision 2026-08-07: no typos, no casing     #
    # noise — surface-noise perturbation is a different family            #
    # (BrittleBench) and out of the paper's scope statement.              #
    # ------------------------------------------------------------------ #
    "colloquial_chatty": (
        "Rewrite the question the way someone would casually ask a friend in "
        "a chat message: relaxed, colloquial, contractions welcome, maybe an "
        "informal opener — but fluent English, no typos or slang so heavy it "
        "obscures the meaning."
    ),
    "verbose_polite": (
        "Rewrite the question as an elaborately polite request in which the "
        "question is embedded in a longer sentence, e.g. 'Would you mind "
        "telling me ...' or 'I would really appreciate it if you could let "
        "me know ...'. Keep every entity and constraint."
    ),
    "cleft_or_passive": (
        "Rewrite the question using a marked syntactic construction: a cleft "
        "('What is the place where ...'), a passive, or fronting. Change the "
        "syntax substantially while keeping the meaning exactly."
    ),
    "indirect_statement": (
        "Rewrite the question as an indirect information-seeking statement "
        "rather than a direct question, e.g. 'I'm trying to find out ...' or "
        "'I need to know ...'. It should still clearly ask for exactly the "
        "same answer."
    ),
    "headline_style": (
        "Rewrite the question in the compressed style of a quiz-show clue or "
        "article headline: punchy, may drop the question word, but every "
        "entity, relation and constraint that fixes the answer must remain."
    ),
    "spoken_conversational": (
        "Rewrite the question the way a person would ask it out loud in "
        "conversation: natural spoken rhythm, may open with 'So' or 'Okay', "
        "may restate the topic first ('That show X — where was it filmed?'). "
        "Fluent speech, no filler stutter."
    ),
    "narrative_context": (
        "Rewrite the question with a one-clause narrative lead-in that adds "
        "NO new facts, e.g. 'I was discussing this yesterday and couldn't "
        "remember: ...'. The lead-in must be generic and the question "
        "complete and unchanged in meaning."
    ),
    "exam_question": (
        "Rewrite the question in the formal style of an exam or quiz sheet, "
        "e.g. 'State the ...' or 'Identify the ...'. Imperative form is "
        "fine; keep every constraint that determines the answer."
    ),
}


_SYSTEM = (
    "You are a careful paraphrase generator. Your task is to rewrite a "
    "single question so the answer set is preserved bit-for-bit. The "
    "rewritten question MUST have the same correct answer (and the same set "
    "of acceptable answers) as the original. You MAY change wording, word "
    "order, syntax, and tone. You MUST NOT add or remove information that "
    "would change which answers are correct. You MUST NOT answer the "
    "question.\n\n"
    "Output exactly one line: the rewritten question. No preamble, no "
    "explanation, no surrounding quotes, no trailing notes."
)


_USER_TEMPLATE = (
    "Persona to write as: {persona}\n\n"
    "Original question:\n"
    "{question}\n\n"
    "Rewritten question (one line, in the persona above, with the same "
    "answer set as the original):"
)


def build_paraphrase_messages(question: str, role: RoleName) -> list[ChatMessage]:
    """Build the (system, user) message pair for one paraphrase request.

    The system prompt is identical across roles; only the persona varies, so
    cache hits are still maximally reused within a question across samples.
    """
    if role not in _PERSONA:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLE_NAMES}")
    persona = _PERSONA[role]
    user_text = _USER_TEMPLATE.format(persona=persona, question=question.strip())
    return [
        ChatMessage(role="system", content=_SYSTEM),
        ChatMessage(role="user", content=user_text),
    ]


def list_persona_descriptions() -> dict[RoleName, str]:
    """Returned for documentation / write-up purposes only."""
    return dict(_PERSONA)


# --------------------------------------------------------------------------- #
# R6 — the generator-width dial (positive control for rho_F)                  #
# --------------------------------------------------------------------------- #

# Each arm is a full generator specification G: instruction set + sampling
# temperature (+ optionally a different generator model). Width is ordered
# BY DESIGN (narrow < medium < wide) and must be CONFIRMED by the manipulation
# check on the accepted universes (scripts/width_dial_analysis.py) — the NLI +
# gold gates are identical across arms and may censor the wide arm harder.
#
# medium == the production configuration that generated data/
# paraphrases_ambigqa.parquet, so the middle dial point reuses the v3 data.
# swap == medium-width instructions under a different-family generator
# (paraphraser-swap ablation; per LITERATURE_INTEGRATION_2026-08-07 §4.4 no
# published prompt-sensitivity metric has one).
WIDTH_ARMS: dict[str, dict] = {
    "narrow": {
        "roles": ["minimal_synonym", "minimal_reorder", "minimal_function_words"],
        "temperature": 0.5,
        "generator_model": None,          # config default (phi_4_14b)
        "judge_model": None,              # config default (phi_4_14b)
        "cache": "data/paraphrases_width_narrow.parquet",
    },
    "medium": {
        "roles": list(ROLE_NAMES),
        "temperature": 0.8,
        "generator_model": None,
        "judge_model": None,
        "cache": "data/paraphrases_ambigqa.parquet",   # the existing v3 universes
    },
    "wide": {
        "roles": ["colloquial_chatty", "verbose_polite", "cleft_or_passive",
                  "indirect_statement", "headline_style", "spoken_conversational",
                  "narrative_context", "exam_question"],
        "temperature": 1.0,
        "generator_model": None,
        "judge_model": None,
        "cache": "data/paraphrases_width_wide.parquet",
    },
    "swap": {
        "roles": list(ROLE_NAMES),        # medium-width instructions ...
        "temperature": 0.8,               # ... at the medium temperature ...
        "generator_model": "olmo_2_13b",  # ... under a different-family generator
        # The gold judge swaps WITH the generator: the production rule is
        # judge == generator (Phi-4 judges Phi-4), and the 40 GB A100 cannot
        # hold OLMo + Phi-4 at once (job-5762430 OOM pattern). The primary
        # semantic gate (DeBERTa NLI) is identical across all arms; per-arm
        # gate-censoring stats are persisted so a strictness difference is
        # visible rather than silent.
        "judge_model": "olmo_2_13b",
        "cache": "data/paraphrases_width_swap.parquet",
    },
}


def resolve_width_arm(arm: str) -> dict:
    """Validated arm spec; raises on unknown arm or roles missing a persona."""
    if arm not in WIDTH_ARMS:
        raise ValueError(f"unknown width arm {arm!r}; expected one of {sorted(WIDTH_ARMS)}")
    spec = dict(WIDTH_ARMS[arm])
    missing = [r for r in spec["roles"] if r not in _PERSONA]
    if missing:
        raise ValueError(f"width arm {arm!r} references personas without prompts: {missing}")
    return spec
