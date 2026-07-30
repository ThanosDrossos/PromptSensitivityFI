# Umbauplan: Pivot zu AmbigQA-Spezifität

Übergabe-Spezifikation für Claude Code. Ausführen **im Repo-Root** `Code/PromptSensitivityFI/`.
Alle Pfade relativ dazu. Prosa deutsch, Code/Identifier englisch (wie im Repo).

---

## 0. Kontext und Ziel

Das Projekt hat bisher Prompt-Sensitivität als FI_in über Paraphrasen gemessen und als
manipulierte Achse einen Kontext- bzw. Reasoning-Ladder (HotpotQA/MuSiQue) benutzt. Diese
Achse misst aber Evidenz/Schwierigkeit, nicht Fragespezifität, und der Reasoning-Ladder leakt
die Lösung. Neue, vom Betreuer abgesegnete Richtung:

- Manipulierte Variable (X-Achse) = **Spezifität der Frage**, operationalisiert über
  **AmbigQA-Disambiguierung** (mehrdeutige Frage -> disambiguierte Frage).
- Drei Zielgrößen bleiben/kommen:
  - **FI_in** (Formulierungs-Sensitivität, über Paraphrasen) — bleibt.
  - **FI_out** (realisierte Ausgabe-Konzentration des Modells, `log2|A_q| - H_sem`) — bleibt, ist schon im `MetricTuple`.
  - **FI_spec** (normative Fragespezifität in Bits, aus AmbigQA-Antwortmengen) — **neu**.
- Kernaussage des Umbaus: **Datensatz + manipulierte Achse tauschen, Metrik-Stack behalten.**

**Nicht-Ziele dieses Umbaus:** kein Neuschreiben der Metrik-Mathematik, kein Löschen der
alten Loader/Ladders, keine VoI-Kontext-Arme (das ist die separat markierte Phase 2).

---

## 1. Kernidee und das Zell-Modell

Die Spezifitäts-Manipulation lebt **im Fragetext**, nicht in Kontext-Paragraphen. Deshalb
bildet jede Spezifitätsstufe auf eine **Closed-Book-Zelle** ab: gleiche Zell-Maschinerie wie
heute, nur mit **leerer Paragraphen-Liste** (der vorhandene Prompt-Assembler erzeugt dann eine
reine Frage ohne Kontextblock, siehe `prompts/templates/qa_prompt.py`, Level-0-Pfad).

Pro Frage gibt es zwei Stufen (v1):

| spec_level | Fragetext | Ziel-Antwort (Gold, fix) | m_valid | FI_spec [bits] |
|---|---|---|---|---|
| 0 (ambig) | die originale, mehrdeutige Frage Q | a_i der Zielinterpretation | m0 | `log2(m0/m0) = 0` |
| 1 (disambig) | die disambiguierte Frage Q_i | a_i | 1 | `log2(m0/1) = log2(m0)` |

**Wichtig (der Guardrail):** Das Scoring-Gold `a_i` ist über beide Stufen **fix**. Nur der
Fragetext (und damit `m_valid`) ändert sich. So driftet die Ground Truth nicht; wir messen, ob
das Modell mit steigender Spezifität die feste Zielantwort öfter trifft.

Erwartete Effekte (Validierungshypothesen, für das Gate):
- accuracy(level 1) >= accuracy(level 0)
- FI_in(level 1) <= FI_in(level 0)
- FI_out(level 1) >= FI_out(level 0), H_sem(level 1) <= H_sem(level 0)
- FI_spec(level 1) > FI_spec(level 0)

Paraphrasen werden **pro Stufe** über den jeweiligen Fragetext gebildet (für FI_in). Die
bestehende NLI-Äquivalenz-Pipeline hält die Spezifität innerhalb einer Stufe konstant.

---

## 2. Was bleibt, was neu ist, was deaktiviert wird

**Unverändert wiederverwenden (nicht anfassen):**
- `models/` (registry, local_hf, cache, embedding, rate_limiter)
- `paraphrases/` (generate, nli_filter, constraint_filter, deduplicate, pipeline)
- `scoring/nli_with_gold.py`
- `metrics/` Mathematik: `fi_in.py`, `h_sem.py`, `fi_out.py`, `errica.py`, `rho_u.py`,
  `ess_in.py`, `spread.py`, `variation_ratio.py`, `posix.py`, `orchestrator.py`
- `config.py`, `logging_setup.py`, `prompts/templates/qa_prompt.py`

**Neu anlegen:**
- `prompt_sensitivity/data/ambigqa_schemas.py` — Schema `AmbigQuestion` + `AmbigInterpretation`
- `prompt_sensitivity/data/load_ambigqa.py` — Loader
- `prompt_sensitivity/specificity/__init__.py`
- `prompt_sensitivity/specificity/build_levels.py` — Stufen-Builder + `SpecRow`
- `prompt_sensitivity/metrics/fi_spec.py` — FI_spec-Metrik
- `prompt_sensitivity/scripts/run_specificity.py` — Treiber (dünn, ruft die e2e-Zell-Helfer)
- `prompt_sensitivity/scripts/smoke_specificity.py` — Verifikations-Gate
- Tests: `tests/test_ambigqa_loader.py`, `tests/test_fi_spec.py`, `tests/test_specificity_builder.py`
- Fixture: `tests/fixtures/ambigqa_sample.json` (2-3 Records, manuell aus dem Release)

**Deaktivieren (nicht löschen, nur im neuen Treiber nicht verwenden):**
- `ladders/*` (Kontext-/Reasoning-Ladder) als primäre Manipulation
- `scoring/chain_score.py` (AmbigQA hat keine Decomposition, binäres NLI-with-gold reicht)
- `data/load_musique.py`, `data/load_hotpotqa.py`, `data/load_2wiki.py` bleiben liegen

---

## 3. Neuer Loader und Datenstruktur

### 3.1 Schema (`data/ambigqa_schemas.py`)

```python
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

class AmbigInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disambiguated_question: str          # der disambiguierte Rewrite Q_i
    answers: list[str] = Field(min_length=1)   # akzeptierte Antwortvarianten fuer a_i

class AmbigQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    dataset: Literal["ambigqa"] = "ambigqa"
    question: str                        # die originale (mehrdeutige) Frage
    interpretations: list[AmbigInterpretation] = Field(min_length=1)

    def m0(self) -> int:
        return len(self.interpretations)

    def is_ambiguous(self) -> bool:
        return self.m0() > 1
```

### 3.2 Loader (`data/load_ambigqa.py`)

Quelle: HuggingFace `ambig_qa`, Config `light`, Split `validation` (Test ist nicht public).
Kanonische Alternative: das GitHub-Release `shmsw25/AmbigQA`.

**Erwartete HF-Struktur (verifizieren, dann mappen):** jeder Record hat `question` und
`annotations`. Eine Annotation ist `type == "singleAnswer"` (eindeutig, `answer: list[str]`)
oder `type == "multipleQAs"` mit `qaPairs: list[{question, answer: list[str]}]`.

Mapping:
- `multipleQAs` -> `AmbigQuestion` mit `interpretations = [AmbigInterpretation(disambiguated_question=qa.question, answers=qa.answer) for qa in qaPairs]`. `m0 = len(qaPairs)`.
- `singleAnswer` -> optionaler Hoch-Spezifitäts-Anker (`m0 = 1`), standardmäßig **verworfen**
  (Filter `min_interpretations = 2`), per Config einschaltbar.

**Auftrag an den Agenten:** die reale HF-Schema-Form einmal per `load_dataset(...).features`
inspizieren und das Parsing defensiv daran ausrichten (wie die bestehenden Loader es tun,
z. B. `load_musique.py` mit seinen Feld-Fallbacks). Nicht auf die exakten Feldnamen oben
verlassen, sondern verifizieren.

Signatur analog zu den anderen Loadern:

```python
def load_ambigqa(*, hf_dataset="ambig_qa", hf_config="light",
                 split="validation", min_interpretations=2,
                 cache_dir=None) -> list[AmbigQuestion]: ...
```

---

## 4. Spezifitäts-Stufen-Builder (`specificity/build_levels.py`)

```python
from pydantic import BaseModel, ConfigDict, Field

class SpecRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    question_id: str
    spec_level: int                 # 0 = ambig, 1 = disambiguiert
    question_text: str              # Q bei level 0, Q_i bei level 1
    target_answers: list[str]       # a_i-Varianten, FIX ueber beide Stufen
    m_valid: int                    # level 0 -> m0, level 1 -> 1
    m0: int
    target_idx: int                 # welche Interpretation als Ziel gewaehlt wurde

def build_spec_levels(q: AmbigQuestion, *, seed: int) -> list[SpecRow]:
    # 1. Zielinterpretation deterministisch waehlen: idx = seeded_choice(range(q.m0()), seed, q.id)
    # 2. level 0: question_text = q.question,           m_valid = q.m0()
    #    level 1: question_text = interp[idx].disambiguated_question, m_valid = 1
    #    target_answers = interp[idx].answers in beiden Faellen
    # 3. gibt [SpecRow(level0), SpecRow(level1)] zurueck
```

Zielwahl deterministisch (Hash aus `q.id + seed`), damit Paraphrasen-Prep und alle
Modell-Läufe dieselbe Zielinterpretation benutzen. Optional später: Zwischenstufen. Für v1
nur 0 und 1.

---

## 5. FI_spec-Metrik (`metrics/fi_spec.py`)

Reine Funktion, keine Modellabhängigkeit:

```python
import math

def fi_spec_bits(m0: int, m_valid: int) -> float:
    """Bits, die die Frage am Antwortraum entfernt (relativ zur vollen Mehrdeutigkeit).
    level 0: m_valid = m0 -> 0.0 ; level 1: m_valid = 1 -> log2(m0)."""
    if m0 <= 0 or m_valid <= 0:
        return 0.0
    return math.log2(m0 / m_valid)
```

Test deckt ab: `fi_spec_bits(4, 4) == 0.0`, `fi_spec_bits(4, 1) == 2.0`, `fi_spec_bits(1, 1) == 0.0`.

---

## 6. Scoring: Mehr-Varianten-Gold

`scoring/nli_with_gold.py::f_score_batch(gold, answers, ...)` nimmt aktuell **ein** Gold-String.
AmbigQA gibt pro Interpretation eine Liste akzeptierter Antworten. Erweiterung (neue Funktion,
alte nicht verändern):

```python
def f_score_batch_multi_gold(golds: list[str], answers, *, config=None, permissive=False) -> list[int]:
    """F=1 wenn die Antwort GEGEN IRGENDEIN Gold aus `golds` besteht (OR ueber Varianten).
    Ruft f_score_batch je Gold und nimmt das elementweise Maximum."""
```

Im Treiber `target_answers` als `golds` durchreichen. `permissive`-Spalte wie gehabt zusätzlich
mitschreiben.

---

## 7. MetricTuple-Erweiterung (`metrics/schemas.py`)

`MetricTuple` ist frozen mit `extra="forbid"`. Neue First-Class-Felder ergänzen (alle optional,
damit alte Parquets weiter laden):

```python
    fi_spec: float | None = None        # log2(m0/m_valid), aus fi_spec_bits
    spec_level: int | None = None       # 0 | 1
    m_valid: int | None = None
    m0: int | None = None
    target_idx: int | None = None
```

`orchestrator.build_metric_tuple(...)` bekommt optionale Parameter `fi_spec`, `spec_level`,
`m_valid`, `m0`, `target_idx` durchgereicht und setzt sie im Tuple. Die Metrik-Mathematik im
Orchestrator bleibt unverändert; FI_spec wird **nicht** dort berechnet, sondern vom Treiber aus
`fi_spec_bits(...)` übergeben (Trennung: Orchestrator = modellseitige Mathematik, FI_spec =
datensatzseitig).

`dataset`, `spec_level` etc. dürfen alternativ wie die bestehenden v6-Spalten **nach**
`model_dump()` im Treiber angehängt werden (Muster siehe `e2e_smoke._run_cell`), falls das
Ändern des frozen Modells vermieden werden soll. Bevorzugt: First-Class-Felder, weil FI_spec
eine Headline-Größe ist.

---

## 8. Treiber (`scripts/run_specificity.py`)

Dünner Treiber, der die **vorhandene** Zell-Maschinerie aus `e2e_smoke.py` wiederverwendet.
Ablauf:

1. `configure_logging("run_specificity")`, `config = load_config()`.
2. Fragen laden: `load_ambigqa(...)`, auf `n_questions` beschränken, nur `is_ambiguous()`.
3. Pro Frage `build_spec_levels(q, seed=config.random_seed)` -> 2 `SpecRow`.
4. Pro `SpecRow`: Paraphrasen-Universe über `question_text` bauen
   (`paraphrases.pipeline.build_paraphrase_set(qid, row.question_text, config,
   gold_answer=row.target_answers[0])`), auf `config.paraphrases.n_per_question` (v1: 10) kappen.
   Cachebar persistieren analog `_generate_musique_paraphrases` (eigene Parquet-Datei
   `data/paraphrases_ambigqa.parquet`).
5. Pro (SpecRow, model): eine **Closed-Book-Zelle** rechnen. Am einfachsten die vorhandene
   `_run_cell`-Logik wiederverwenden, indem ein minimaler `LadderRow`-Adapter gebaut wird:
   `LadderRow(question_id=row.question_id, ladder_type="random", ladder_family="context",
   level_idx=row.spec_level, level=row.spec_level, paragraph_indices=[], paragraph_titles=[],
   gold_count=0)`. Leere `paragraph_indices` => `_assemble_messages` erzeugt den Closed-Book-Prompt.
   - Das Fragen-Objekt, das `_run_cell` erwartet, hat kein `question_decomposition` =>
     `has_decomposition()` ist False => automatisch binärer NLI-with-gold-Pfad (kein CoT,
     kein chain_score). Gold = `row.target_answers` (Multi-Gold, Abschnitt 6).
   - Falls `_run_cell` zu eng an `MultiHopQuestion` gekoppelt ist: einen leichten Adapter
     `_SpecQuestionView` bauen, der `id`, `question`, `answer`(=target_answers[0]),
     `dataset="ambigqa"`, `paragraphs=[]`, `has_decomposition()->False`, `n_hops=None`,
     `question_decomposition=[]` liefert. Nichts an `MultiHopQuestion` selbst ändern.
6. Nach der Zelle: `fi_spec_bits(row.m0, row.m_valid)` berechnen und zusammen mit
   `spec_level, m_valid, m0, target_idx, dataset="ambigqa"` an die Row hängen.
7. Checkpoint pro Zelle (Muster `e2e_smoke._checkpoint`), Resume über
   `(question_id, spec_level, model_key)`.
8. Ausgabe: `data/specificity_metrics.parquet`.

**Wiederverwendete Helfer aus `e2e_smoke.py` (importieren, nicht kopieren):** `_sample_response`,
`_clustering_inputs`, `_assemble_messages`, plus `metrics.build_metric_tuple`,
`models.embedding.encode_texts`, `metrics.h_sem.cluster_responses_pooled`. Wenn diese Helfer
`MultiHopQuestion` erwarten, deckt der `_SpecQuestionView`-Adapter das ab.

CLI-Flags: `--n-questions` (default 50), `--models` (default ein lokales Modell),
`--k-samples` (default `config.h_sem.n_samples_per_prompt`), `--max-paraphrases` (default 10),
`--fast` (H_sem/Embeddings/POSIX aus, nur FI_in + accuracy + FI_spec), `--out`, `--dry-run`.

---

## 9. Config-Änderungen (`config.yaml`)

```yaml
sampling:
  ambigqa:
    hf_dataset: "ambig_qa"
    hf_config: "light"
    split: validation
    n_questions: 50
    min_interpretations: 2        # nur mehrdeutige (multipleQAs) Records
    include_single_answer_anchor: false

specificity:
  levels: [0, 1]                  # v1: ambig, disambiguiert
  target_seed: 42                 # deterministische Zielinterpretation

paraphrases:
  n_per_question: 10              # v1 heruntergesetzt (war 30) fuer Kosten; FI_in bleibt messbar
```

`config.py`: passende frozen Pydantic-Blöcke `AmbigQASamplingConfig` und `SpecificityConfig`
ergänzen, beide optional (`| None = None` bzw. Defaults), damit bestehende Configs weiter
validieren.

---

## 10. Tests (Muster: `tests/` spiegelt Module)

- `test_ambigqa_loader.py`: Fixture parsen; `m0` korrekt; `is_ambiguous`; singleAnswer wird
  bei `min_interpretations=2` verworfen; Antwortvarianten landen als Liste.
- `test_fi_spec.py`: die Bit-Fälle aus Abschnitt 5.
- `test_specificity_builder.py`: 2 Rows; `target_answers` in beiden identisch; `m_valid`
  {level0: m0, level1: 1}; Zielwahl deterministisch bei fixem Seed.
- `test_scoring.py` erweitern: `f_score_batch_multi_gold` = OR über Varianten.
- `test_metric_tuple.py` erweitern: neue Felder default None, setzbar.

Alle heavy Seams (DeBERTa, Modell-Load) wie im Repo patchbar halten, damit die Tests ohne GPU
laufen.

---

## 11. Verifikations-Gate (`scripts/smoke_specificity.py`)

Smoke: ~5 AmbigQA-Fragen × 2 Stufen × 1 lokales Modell, `--max-paraphrases 6 --k-samples 3`.
Asserts (aggregiert über die Fragen):
- Spalten vorhanden: `fi_spec`, `spec_level`, `m_valid`, `f_mean`, `aufi_in` (FI_in),
  `fi_out_mean`, `h_sem_mean`.
- `mean(fi_spec[level=1]) > mean(fi_spec[level=0])`.
- `mean(f_mean[level=1]) >= mean(f_mean[level=0])`.
- Richtungscheck (nur Warnung, kein Hard-Fail bei kleinem N): `FI_in[level=1] <= FI_in[level=0]`.
- Parquet lädt sauber zurück.

Exit != 0, wenn ein Hard-Assert bricht.

---

## 12. Do-not-touch / Scope-Guard

- Metrik-Mathematik in `metrics/fi_in.py`, `h_sem.py`, `fi_out.py`, `errica.py`, `rho_u.py`,
  `ess_in.py`, `posix.py` **nicht** ändern.
- `MultiHopQuestion` **nicht** ändern (Adapter benutzen).
- `ladders/`, `data/load_musique.py|load_hotpotqa.py|load_2wiki.py`, `scoring/chain_score.py`
  **nicht** löschen, nur nicht verwenden.
- Modell-, Cache-, Paraphrasen-Layer bleiben unverändert.
- Keine VoI-Kontext-Arme in dieser Runde (Phase 2).

---

## 13. Umsetzungsreihenfolge

1. `data/ambigqa_schemas.py` + `data/load_ambigqa.py` + Fixture + `test_ambigqa_loader.py`.
2. `metrics/fi_spec.py` + `test_fi_spec.py`.
3. `specificity/build_levels.py` + `test_specificity_builder.py`.
4. `scoring/nli_with_gold.py`: `f_score_batch_multi_gold` + Test.
5. `metrics/schemas.py` + `orchestrator.py`: neue Felder + Durchreichen.
6. `config.yaml` + `config.py`: Blöcke `ambigqa` und `specificity`.
7. `scripts/run_specificity.py` (Adapter + Zell-Reuse).
8. `scripts/smoke_specificity.py` (Gate) + einmal ausführen.
9. Analyse: kleine Variante von `scripts/show_results.py`, die auf `spec_level` pivotiert und
   `f_mean`, `aufi_in` (FI_in), `fi_out_mean`, `fi_spec` je Stufe nebeneinanderstellt.

Nach jedem Schritt `make test` grün halten.

---

## 14. Phase 2 (später, NICHT jetzt umsetzen)

VoI-Kontext-Arme: pro Frage {disambiguierend-nicht-aufdeckend, teilrelevant, Distraktor,
irreführend} als kurzer Kontext-Zusatz, geerntet aus den AmbigQA-Interpretations-Klauseln plus
einem automatischen NLI-Nicht-Aufdeckungs-Gate (`NLI(c ⊨ a*)` niedrig). Nutzt dieselbe
Closed-Book-Zelle mit einer zusätzlichen Kontextzeile. Erst nach grünem v1-Gate anfassen.
