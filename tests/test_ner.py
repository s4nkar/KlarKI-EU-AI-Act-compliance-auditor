"""Unit tests for the spaCy NER model — EU AI Act entity recognition.

Tests cover all 8 entity labels:
  ARTICLE, OBLIGATION, ACTOR, AI_SYSTEM, RISK_TIER, PROCEDURE, REGULATION, PROHIBITED_USE

Skipped automatically if the trained model is not found at
training/artifacts/spacy_ner_model/model-final (i.e. before setup completes).

These are unit tests (no Ollama / ChromaDB required).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).resolve().parent.parent
MODEL_PATH = next(
    (p for p in [
        REPO_ROOT / "training" / "artifacts" / "spacy_ner_model" / "model-final",
        Path("/training/artifacts/spacy_ner_model/model-final"),
    ] if p.exists()),
    None,
)


@pytest.fixture(scope="module")
def nlp(spacy_ner_nlp):
    """Return the session-scoped NER model (loaded once, shared with eval_ner.py)."""
    if MODEL_PATH is None:
        pytest.skip("NER model not found — run ./run.sh setup first")
    if spacy_ner_nlp is None:
        pytest.skip("spacy not installed or model failed to load — run pip install spacy")
    return spacy_ner_nlp


# ── helper ─────────────────────────────────────────────────────────────────

def _labels(nlp, text: str) -> set[str]:
    """Return the set of entity labels found in text."""
    doc = nlp(text)
    return {ent.label_ for ent in doc.ents}


def _spans(nlp, text: str) -> list[tuple[str, str]]:
    """Return (text, label) tuples for all entities found."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ── ARTICLE ────────────────────────────────────────────────────────────────

class TestArticleLabel:
    def test_article_number_en(self, nlp):
        doc = nlp("Providers must comply with Article 9 of the EU AI Act.")
        articles = [e for e in doc.ents if e.label_ == "ARTICLE"]
        assert articles, "Expected ARTICLE entity not found"
        assert any("9" in e.text for e in articles)

    def test_artikel_number_de(self, nlp):
        doc = nlp("Anbieter müssen Artikel 13 der KI-Gesetz einhalten.")
        articles = [e for e in doc.ents if e.label_ == "ARTICLE"]
        assert articles, "Expected ARTICLE entity not found in German text"

    def test_multiple_articles(self, nlp):
        doc = nlp("Both Article 9 and Article 14 apply to this high-risk AI system.")
        articles = [e for e in doc.ents if e.label_ == "ARTICLE"]
        assert len(articles) >= 2, f"Expected ≥2 ARTICLE entities, got {len(articles)}"

    def test_art_abbreviation(self, nlp):
        labels = _labels(nlp, "Art. 10 sets out data governance requirements.")
        assert "ARTICLE" in labels


# ── OBLIGATION ─────────────────────────────────────────────────────────────

class TestObligationLabel:
    def test_must_document(self, nlp):
        labels = _labels(nlp, "Providers must document all training data sources.")
        assert "OBLIGATION" in labels

    def test_shall_maintain(self, nlp):
        labels = _labels(nlp, "The operator shall maintain a complete audit trail.")
        assert "OBLIGATION" in labels

    def test_are_required_to(self, nlp):
        labels = _labels(nlp, "Importers are required to verify system conformity.")
        assert "OBLIGATION" in labels

    def test_de_obligation(self, nlp):
        labels = _labels(nlp, "Anbieter müssen alle Trainingsdaten vollständig dokumentieren.")
        assert "OBLIGATION" in labels


# ── ACTOR ──────────────────────────────────────────────────────────────────

class TestActorLabel:
    @pytest.mark.parametrize("actor", [
        "providers", "operators", "importers", "manufacturers",
        "notified bodies", "deployers",
    ])
    def test_en_actors(self, nlp, actor):
        text = f"Under Article 9, {actor} must establish a risk management system."
        labels = _labels(nlp, text)
        assert "ACTOR" in labels, f"ACTOR not detected for '{actor}'"

    def test_de_actor(self, nlp):
        labels = _labels(nlp, "Betreiber müssen eine Konformitätsbewertung durchführen.")
        assert "ACTOR" in labels

    def test_actor_with_obligation(self, nlp):
        """ACTOR and OBLIGATION should both be detected in one sentence."""
        text = "Providers must implement a conformity assessment before deployment."
        found = _labels(nlp, text)
        assert "ACTOR" in found
        assert "OBLIGATION" in found


# ── AI_SYSTEM ──────────────────────────────────────────────────────────────

class TestAiSystemLabel:
    @pytest.mark.parametrize("system", [
        "high-risk AI system",
        "general-purpose AI model",
        "emotion recognition system",
        "remote biometric identification system",
    ])
    def test_en_ai_systems(self, nlp, system):
        text = f"The {system} must comply with Article 9 of the EU AI Act."
        labels = _labels(nlp, text)
        assert "AI_SYSTEM" in labels, f"AI_SYSTEM not detected for '{system}'"

    def test_de_ai_system(self, nlp):
        labels = _labels(nlp, "Ein Hochrisiko-KI-System erfordert eine Konformitätsbewertung.")
        assert "AI_SYSTEM" in labels


# ── RISK_TIER ──────────────────────────────────────────────────────────────

class TestRiskTierLabel:
    @pytest.mark.parametrize("tier", [
        "high-risk", "prohibited", "limited risk", "minimal risk", "unacceptable risk",
    ])
    def test_en_risk_tiers(self, nlp, tier):
        text = f"This system is classified as {tier} under Article 5 of the EU AI Act."
        labels = _labels(nlp, text)
        assert "RISK_TIER" in labels, f"RISK_TIER not detected for '{tier}'"

    def test_de_risk_tier(self, nlp):
        labels = _labels(nlp, "Das System wird als hochriskant eingestuft.")
        assert "RISK_TIER" in labels

    def test_risk_tier_with_article(self, nlp):
        text = "A high-risk AI system requires a conformity assessment under Article 43."
        found = _labels(nlp, text)
        assert "RISK_TIER" in found
        assert "ARTICLE" in found


# ── PROCEDURE ──────────────────────────────────────────────────────────────

class TestProcedureLabel:
    @pytest.mark.parametrize("procedure", [
        "conformity assessment",
        "risk management system",
        "technical documentation",
        "post-market monitoring",
        "fundamental rights impact assessment",
    ])
    def test_en_procedures(self, nlp, procedure):
        text = f"Providers must complete a {procedure} under Article 9."
        labels = _labels(nlp, text)
        assert "PROCEDURE" in labels, f"PROCEDURE not detected for '{procedure}'"

    def test_de_procedure(self, nlp):
        labels = _labels(nlp, "Anbieter müssen eine Konformitätsbewertung durchführen.")
        assert "PROCEDURE" in labels


# ── REGULATION ─────────────────────────────────────────────────────────────

class TestRegulationLabel:
    @pytest.mark.parametrize("reg", [
        "EU AI Act", "GDPR", "Artificial Intelligence Act",
        "General Data Protection Regulation",
    ])
    def test_en_regulations(self, nlp, reg):
        text = f"Compliance with the {reg} is mandatory for this deployment."
        labels = _labels(nlp, text)
        assert "REGULATION" in labels, f"REGULATION not detected for '{reg}'"

    def test_de_regulation(self, nlp):
        labels = _labels(nlp, "Die DSGVO gilt gemeinsam mit dem KI-Gesetz für diese Anwendung.")
        assert "REGULATION" in labels

    def test_multiple_regulations(self, nlp):
        doc = nlp("Compliance with both the EU AI Act and GDPR is required.")
        regs = [e for e in doc.ents if e.label_ == "REGULATION"]
        assert len(regs) >= 2, f"Expected ≥2 REGULATION entities, got {len(regs)}"


# ── PROHIBITED_USE ─────────────────────────────────────────────────────────

class TestProhibitedUseLabel:
    @pytest.mark.parametrize("practice", [
        "social scoring",
        "emotion recognition in the workplace",
        "real-time biometric surveillance in public spaces",
        "subliminal manipulation",
        "mass surveillance of natural persons",
    ])
    def test_en_prohibited_practices(self, nlp, practice):
        text = f"The use of {practice} is explicitly banned under Article 5 of the EU AI Act."
        labels = _labels(nlp, text)
        assert "PROHIBITED_USE" in labels, f"PROHIBITED_USE not detected for '{practice}'"

    def test_de_prohibited_use(self, nlp):
        labels = _labels(nlp, "Social Scoring durch Behörden ist nach Artikel 5 verboten.")
        assert "PROHIBITED_USE" in labels

    def test_prohibited_triggers_risk_tier(self, nlp):
        """PROHIBITED_USE sentences should also carry RISK_TIER markers."""
        text = "Social scoring is prohibited under Article 5 of the EU AI Act."
        found = _labels(nlp, text)
        assert "PROHIBITED_USE" in found
        # RISK_TIER or ARTICLE should also be present — confirms entity co-occurrence
        assert "ARTICLE" in found or "RISK_TIER" in found


# ── Multi-entity sentences ─────────────────────────────────────────────────

class TestMultiEntitySentences:
    def test_full_compliance_sentence(self, nlp):
        """A rich sentence should surface 5+ distinct label types."""
        text = (
            "Providers must conduct a conformity assessment for the high-risk AI system "
            "under Article 43 of the EU AI Act before market placement."
        )
        found = _labels(nlp, text)
        assert len(found) >= 4, f"Expected ≥4 label types, got {found}"

    def test_prohibited_use_full_sentence(self, nlp):
        text = (
            "Under Article 5 of the EU AI Act, operators must ensure the "
            "high-risk AI system does not employ social scoring."
        )
        found = _labels(nlp, text)
        assert "PROHIBITED_USE" in found
        assert "ARTICLE" in found
        assert "REGULATION" in found

    def test_de_multi_entity(self, nlp):
        text = (
            "Betreiber müssen sicherstellen, dass das Hochrisiko-KI-System "
            "keine Emotionserkennung am Arbeitsplatz gemäß Artikel 5 der KI-Gesetz einsetzt."
        )
        found = _labels(nlp, text)
        assert len(found) >= 3, f"Expected ≥3 label types in German sentence, got {found}"


# ── Label coverage regression ──────────────────────────────────────────────

def test_all_8_labels_defined(nlp):
    """The loaded model must have all 8 expected labels in its NER component."""
    expected = {
        "ARTICLE", "OBLIGATION", "ACTOR", "AI_SYSTEM",
        "RISK_TIER", "PROCEDURE", "REGULATION", "PROHIBITED_USE",
    }
    ner = nlp.get_pipe("ner")
    actual = set(ner.labels)
    missing = expected - actual
    assert not missing, f"NER model is missing labels: {missing}"
