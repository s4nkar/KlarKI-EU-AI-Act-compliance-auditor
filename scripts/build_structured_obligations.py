"""Deterministic obligation extractor — no LLM, no HTTP calls.

Produces two output files, one per consumer in the inference pipeline:

  data/obligations/eu_ai_act/article_6_annex_iii.jsonl
    → consumed by applicability_engine.py
    → keyed by annex_category int (1–8, matching AnnexIIICategory enum)
    → populates AnnexIIIMatch.obligation_id and .evidence_required

  data/obligations/eu_ai_act/articles_9_15.jsonl
    → consumed by evidence_mapper.py
    → obligations extracted verbatim from Articles 9–15
    → actor values match ActorType.value exactly

  data/obligations/eu_ai_act/article_5.jsonl
    → consumed by evidence_mapper.py (only when applicable_articles=[5])
    → obligation documentation requirements for Art 5(1)(d)/(e) exception cases

All evidence_required values are keys from evidence_mapper._EVIDENCE_SYNONYMS.
All actor values are exactly: "provider", "deployer", "importer", "distributor".
Requirements are verbatim or minimal paraphrase of the actual legal text.
Source paragraph is recorded in each entry for traceability.
"""

import argparse
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "obligations" / "eu_ai_act"
GDPR_OUT_DIR = Path(__file__).parent.parent / "data" / "obligations" / "gdpr"

# ── Article 6 + Annex III — for applicability_engine.py ──────────────────────
# Keyed by annex_category int (1–8, AnnexIIICategory enum values).
# Evidence terms must match evidence_mapper._EVIDENCE_SYNONYMS keys.
# Source: EU AI Act Annex III (as amended by Regulation (EU) 2024/1689)
_ANNEX_III_OBLIGATIONS: list[dict] = [
    {
        "id": "AIACT_ART6_ANNEXIII_CAT1",
        "annex_category": 1,
        "regulation": "eu_ai_act",
        "title": "Biometric identification and categorisation of natural persons",
        "source_paragraph": "Annex III §1",
        "evidence_required": [
            "technical documentation",
            "conformity assessment record",
            "fundamental rights impact assessment",
            "logging and audit trail",
            "human oversight procedure",
        ],
    },
    {
        "id": "AIACT_ART6_ANNEXIII_CAT2",
        "annex_category": 2,
        "regulation": "eu_ai_act",
        "title": "Management and operation of critical infrastructure",
        "source_paragraph": "Annex III §2",
        "evidence_required": [
            "technical documentation",
            "safety validation records",
            "risk management system",
            "mitigation controls",
            "logging and audit trail",
        ],
    },
    {
        "id": "AIACT_ART6_ANNEXIII_CAT3",
        "annex_category": 3,
        "regulation": "eu_ai_act",
        "title": "Education and vocational training",
        "source_paragraph": "Annex III §3",
        "evidence_required": [
            "technical documentation",
            "bias audit",
            "transparency notice",
            "human oversight procedure",
            "appeals mechanism",
        ],
    },
    {
        "id": "AIACT_ART6_ANNEXIII_CAT4",
        "annex_category": 4,
        "regulation": "eu_ai_act",
        "title": "Employment, workers management and access to self-employment",
        "source_paragraph": "Annex III §4",
        "evidence_required": [
            "technical documentation",
            "bias audit",
            "transparency notice",
            "human oversight procedure",
            "worker notification record",
        ],
    },
    {
        "id": "AIACT_ART6_ANNEXIII_CAT5",
        "annex_category": 5,
        "regulation": "eu_ai_act",
        "title": "Access to and enjoyment of essential private and public services",
        "source_paragraph": "Annex III §5",
        "evidence_required": [
            "technical documentation",
            "bias audit",
            "fundamental rights impact assessment",
            "human oversight procedure",
            "appeals mechanism",
        ],
    },
    {
        "id": "AIACT_ART6_ANNEXIII_CAT6",
        "annex_category": 6,
        "regulation": "eu_ai_act",
        "title": "Law enforcement",
        "source_paragraph": "Annex III §6",
        "evidence_required": [
            "technical documentation",
            "fundamental rights impact assessment",
            "logging and audit trail",
            "human oversight procedure",
            "supervisory authority notification",
        ],
    },
    {
        "id": "AIACT_ART6_ANNEXIII_CAT7",
        "annex_category": 7,
        "regulation": "eu_ai_act",
        "title": "Migration, asylum and border control management",
        "source_paragraph": "Annex III §7",
        "evidence_required": [
            "technical documentation",
            "fundamental rights impact assessment",
            "human oversight procedure",
            "logging and audit trail",
            "supervisory authority notification",
        ],
    },
    {
        "id": "AIACT_ART6_ANNEXIII_CAT8",
        "annex_category": 8,
        "regulation": "eu_ai_act",
        "title": "Administration of justice and democratic processes",
        "source_paragraph": "Annex III §8",
        "evidence_required": [
            "technical documentation",
            "fundamental rights impact assessment",
            "human oversight procedure",
            "judicial independence safeguard documentation",
        ],
    },
]

# ── Articles 9–15 — for evidence_mapper.py ───────────────────────────────────
# Requirements are verbatim or minimal paraphrase of the actual article text.
# Actors: exactly "provider" | "deployer" | "importer" | "distributor".
# Evidence terms: keys from evidence_mapper._EVIDENCE_SYNONYMS.
_ARTICLES_9_15_OBLIGATIONS: list[dict] = [

    # ── Article 9 — Risk Management System ───────────────────────────────────
    {
        "id": "AIACT_ART9_001",
        "regulation": "eu_ai_act",
        "article": "Article 9",
        "source_paragraph": "9(1)",
        "title": "Risk management system — establish, implement, document and maintain",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "A risk management system shall be established, implemented, documented "
            "and maintained in relation to high-risk AI systems."
        ),
        "evidence_required": ["risk management system", "technical documentation"],
        "severity": "critical",
        "linked_articles": ["Article 11", "Annex IV"],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART9_002",
        "regulation": "eu_ai_act",
        "article": "Article 9",
        "source_paragraph": "9(2)(a)",
        "title": "Risk identification and analysis",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "Identify and analyse the known and foreseeable risks that the high-risk AI "
            "system can pose to health, safety or fundamental rights when used in "
            "accordance with its intended purpose."
        ),
        "evidence_required": ["risk register"],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART9_003",
        "regulation": "eu_ai_act",
        "article": "Article 9",
        "source_paragraph": "9(2)(b)-(c)",
        "title": "Risk estimation, evaluation and post-market monitoring",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "Estimate and evaluate the risks that may emerge when the system is used "
            "in accordance with its intended purpose and under conditions of reasonably "
            "foreseeable misuse; evaluate risks arising on the basis of data gathered "
            "from the post-market monitoring system referred to in Article 72."
        ),
        "evidence_required": ["risk register", "monitoring procedure", "incident monitoring procedure"],
        "severity": "high",
        "linked_articles": ["Article 72"],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART9_004",
        "regulation": "eu_ai_act",
        "article": "Article 9",
        "source_paragraph": "9(2)(d)",
        "title": "Risk management measures — adoption",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "Adopt appropriate and targeted risk management measures; residual risk "
            "associated with each hazard and the overall residual risk of the high-risk "
            "AI system must be judged acceptable."
        ),
        "evidence_required": ["mitigation controls", "risk register"],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART9_005",
        "regulation": "eu_ai_act",
        "article": "Article 9",
        "source_paragraph": "9(5)-(6)",
        "title": "Testing before placement on market",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "High-risk AI systems shall be tested before being placed on the market or "
            "put into service to identify the most appropriate risk management measures "
            "and to ensure they perform consistently for their intended purpose."
        ),
        "evidence_required": ["safety validation records"],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },

    # ── Article 10 — Data and Data Governance ─────────────────────────────────
    {
        "id": "AIACT_ART10_001",
        "regulation": "eu_ai_act",
        "article": "Article 10",
        "source_paragraph": "10(1)-(2)",
        "title": "Data governance and management practices",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "Training, validation and testing data sets shall be subject to appropriate "
            "data governance and management practices concerning design choices, data "
            "collection processes and origin, data preparation operations, relevant "
            "assumptions, availability and suitability of data sets."
        ),
        "evidence_required": ["data governance documentation"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART10_002",
        "regulation": "eu_ai_act",
        "article": "Article 10",
        "source_paragraph": "10(2)(f)-(g)",
        "title": "Bias examination and mitigation",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "Examine datasets in view of possible biases that are likely to affect "
            "health or safety of persons, lead to prohibited discrimination, or affect "
            "fundamental rights; adopt appropriate measures to detect, prevent and "
            "mitigate possible biases."
        ),
        "evidence_required": ["bias audit", "data governance documentation"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART10_003",
        "regulation": "eu_ai_act",
        "article": "Article 10",
        "source_paragraph": "10(3)",
        "title": "Data quality — relevance, representativeness and completeness",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "Training, validation and testing data sets shall be relevant, sufficiently "
            "representative, and to the best extent possible, free of errors and complete "
            "in view of the intended purpose. They shall have the appropriate statistical "
            "properties, including as regards the persons or groups in relation to whom "
            "the high-risk AI system is intended to be used."
        ),
        "evidence_required": ["data governance documentation", "bias audit"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },

    # ── Article 11 — Technical Documentation ──────────────────────────────────
    {
        "id": "AIACT_ART11_001",
        "regulation": "eu_ai_act",
        "article": "Article 11",
        "source_paragraph": "11(1)",
        "title": "Technical documentation — draw up before market and keep up-to-date",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "The technical documentation of a high-risk AI system shall be drawn up "
            "before that system is placed on the market or put into service and shall "
            "be kept up-to-date."
        ),
        "evidence_required": ["technical documentation"],
        "severity": "critical",
        "linked_articles": ["Annex IV"],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART11_002",
        "regulation": "eu_ai_act",
        "article": "Article 11",
        "source_paragraph": "11(2)",
        "title": "Technical documentation — demonstrate compliance and Annex IV content",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "The technical documentation shall demonstrate that the high-risk AI system "
            "complies with the requirements of Chapter III Section 2 and provide national "
            "competent authorities and notified bodies with all necessary information to "
            "assess compliance. It shall contain at minimum the elements set out in "
            "Annex IV, including: general system description, development process, "
            "monitoring and control information, risk management description, lifecycle "
            "changes, harmonised standards applied, EU declaration of conformity, and "
            "post-market performance evaluation system."
        ),
        "evidence_required": ["technical documentation", "conformity assessment"],
        "severity": "critical",
        "linked_articles": ["Annex IV", "Article 47"],
        "penalty_relevance": True,
    },

    # ── Article 12 — Record-Keeping ───────────────────────────────────────────
    {
        "id": "AIACT_ART12_001",
        "regulation": "eu_ai_act",
        "article": "Article 12",
        "source_paragraph": "12(1)-(3)",
        "title": "Automatic logging capability — technical design requirement",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "High-risk AI systems shall technically allow for the automatic recording "
            "of events (logs) over the lifetime of the system. Logging capabilities "
            "shall ensure a level of traceability of the AI system's functioning "
            "throughout its lifetime that is appropriate to the intended purpose, and "
            "shall enable monitoring of operation with respect to risks and substantial "
            "modifications."
        ),
        "evidence_required": ["logging and audit trail"],
        "severity": "high",
        "linked_articles": ["Article 72"],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART12_002",
        "regulation": "eu_ai_act",
        "article": "Article 12",
        "source_paragraph": "12(5)",
        "title": "Log retention — deployer obligation (minimum six months)",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "The deployer shall retain the logs generated by the high-risk AI system "
            "in so far as such logs are under its control, for a period appropriate to "
            "the intended purpose, of at least six months, unless provided otherwise in "
            "applicable Union or national law."
        ),
        "evidence_required": ["logging and audit trail"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },

    # ── Article 13 — Transparency and Provision of Information to Deployers ───
    {
        "id": "AIACT_ART13_001",
        "regulation": "eu_ai_act",
        "article": "Article 13",
        "source_paragraph": "13(1)",
        "title": "Transparency — design for interpretable operation",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "High-risk AI systems shall be designed and developed in such a way to "
            "ensure that their operation is sufficiently transparent to enable deployers "
            "to interpret the system's output and use it appropriately."
        ),
        "evidence_required": ["transparency notice", "explainability mechanism"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART13_002",
        "regulation": "eu_ai_act",
        "article": "Article 13",
        "source_paragraph": "13(2)-(3)",
        "title": "Instructions for use — content requirements",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "High-risk AI systems shall be accompanied by instructions for use in an "
            "appropriate digital format including: identity and contact details of the "
            "provider; characteristics, capabilities and limitations; accuracy, "
            "robustness and cybersecurity metrics; known risks to health, safety or "
            "fundamental rights; human oversight measures; and a description of logging "
            "mechanisms."
        ),
        "evidence_required": ["technical documentation", "transparency notice"],
        "severity": "high",
        "linked_articles": ["Article 14", "Article 12", "Article 15"],
        "penalty_relevance": True,
    },

    # ── Article 14 — Human Oversight ──────────────────────────────────────────
    {
        "id": "AIACT_ART14_001",
        "regulation": "eu_ai_act",
        "article": "Article 14",
        "source_paragraph": "14(1)-(2)",
        "title": "Human oversight — design and development requirement",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "High-risk AI systems shall be designed and developed, including with "
            "appropriate human-machine interface tools, so that they can be effectively "
            "overseen by natural persons during the period in which the AI system is "
            "in use. Oversight measures shall be proportionate to the risks, level of "
            "autonomy and context of use."
        ),
        "evidence_required": ["human oversight procedure"],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART14_002",
        "regulation": "eu_ai_act",
        "article": "Article 14",
        "source_paragraph": "14(3)",
        "title": "Human oversight — operational capabilities for oversight persons",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "Natural persons assigned human oversight shall be enabled to: fully "
            "understand the capacities and limitations of the system and monitor its "
            "operation; remain aware of automation bias; correctly interpret outputs; "
            "decide not to use the system or override its output; and intervene on or "
            "interrupt the system through a stop procedure."
        ),
        "evidence_required": ["human oversight procedure", "explainability mechanism"],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART14_003",
        "regulation": "eu_ai_act",
        "article": "Article 14",
        "source_paragraph": "14(5)",
        "title": "Human oversight — competence, training and authority",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "Natural persons to whom human oversight is assigned shall have the "
            "necessary competence, training and authority, as well as the necessary "
            "support, to carry out that role."
        ),
        "evidence_required": ["human oversight procedure"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },

    # ── Article 15 — Accuracy, Robustness and Cybersecurity ───────────────────
    {
        "id": "AIACT_ART15_001",
        "regulation": "eu_ai_act",
        "article": "Article 15",
        "source_paragraph": "15(1)-(2)",
        "title": "Accuracy — appropriate level and lifecycle consistency",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "High-risk AI systems shall achieve an appropriate level of accuracy and "
            "perform consistently throughout their lifecycle. Accuracy levels and "
            "relevant accuracy metrics shall be declared in the accompanying "
            "instructions for use."
        ),
        "evidence_required": ["safety validation records", "technical documentation"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART15_002",
        "regulation": "eu_ai_act",
        "article": "Article 15",
        "source_paragraph": "15(3)",
        "title": "Robustness — resilience against errors, faults and inconsistencies",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "High-risk AI systems shall be resilient as regards errors, faults or "
            "inconsistencies that may occur within the system or the environment in "
            "which the system operates. Technical robustness may be achieved through "
            "redundancy solutions including backup or fail-safe plans."
        ),
        "evidence_required": ["safety validation records", "mitigation controls"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART15_003",
        "regulation": "eu_ai_act",
        "article": "Article 15",
        "source_paragraph": "15(5)-(6)",
        "title": "Cybersecurity — resilience against adversarial attacks",
        "actor": ["provider"],
        "requirement_type": "mandatory",
        "requirement": (
            "High-risk AI systems shall be resilient against attempts by third parties "
            "to alter their use, outputs or performance by exploiting system "
            "vulnerabilities, including adversarial attacks, data poisoning attacks, "
            "model evasion, and confidentiality attacks. Technical solutions shall "
            "include measures to prevent, detect, respond to, resolve and control "
            "for attacks."
        ),
        "evidence_required": ["safety validation records", "mitigation controls"],
        "severity": "high",
        "linked_articles": [],
        "penalty_relevance": True,
    },
]

# ── Article 5 — for evidence_mapper.py (applicable_articles=[5] path) ────────
# These represent documentation obligations for the narrow exceptions permitted
# under Article 5(1)(d) and (e). A system with applicable_articles=[5] is
# flagged prohibited; these entries record what authorisation artefacts must
# exist for any permitted exception use.
_ARTICLE_5_OBLIGATIONS: list[dict] = [
    {
        "id": "AIACT_ART5_001",
        "regulation": "eu_ai_act",
        "article": "Article 5",
        "source_paragraph": "5(1)(a)",
        "title": "Prohibition — subliminal techniques AI system",
        "actor": ["provider", "deployer"],
        "requirement_type": "prohibition",
        "requirement": (
            "Placing on the market, putting into service or using an AI system that "
            "deploys subliminal techniques beyond a person's consciousness in order to "
            "materially distort behaviour in a manner that causes or is likely to cause "
            "harm is prohibited."
        ),
        "evidence_required": ["risk register", "fundamental rights impact assessment"],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART5_002",
        "regulation": "eu_ai_act",
        "article": "Article 5",
        "source_paragraph": "5(1)(b)",
        "title": "Prohibition — AI exploiting vulnerabilities of specific groups",
        "actor": ["provider", "deployer"],
        "requirement_type": "prohibition",
        "requirement": (
            "Placing on the market, putting into service or using an AI system that "
            "exploits vulnerabilities of a specific group of persons due to age, "
            "disability or social or economic situation in order to materially distort "
            "behaviour is prohibited."
        ),
        "evidence_required": ["risk register", "fundamental rights impact assessment"],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART5_003",
        "regulation": "eu_ai_act",
        "article": "Article 5",
        "source_paragraph": "5(1)(c)",
        "title": "Prohibition — social scoring by public authorities",
        "actor": ["deployer"],
        "requirement_type": "prohibition",
        "requirement": (
            "The placing on the market or use of AI systems for social scoring by "
            "public authorities that lead to detrimental treatment of individuals or "
            "groups is prohibited."
        ),
        "evidence_required": ["risk register", "fundamental rights impact assessment"],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART5_004",
        "regulation": "eu_ai_act",
        "article": "Article 5",
        "source_paragraph": "5(1)(d)",
        "title": "Restricted use — real-time remote biometric identification in public spaces",
        "actor": ["deployer"],
        "requirement_type": "conditional",
        "requirement": (
            "The use of real-time remote biometric identification systems in publicly "
            "accessible spaces for the purpose of law enforcement is prohibited except "
            "in strictly defined circumstances permitted by law. Where an exception "
            "applies, prior authorisation and documented legal basis are required."
        ),
        "evidence_required": [
            "risk register",
            "supervisory authority notification",
            "fundamental rights impact assessment",
            "logging and audit trail",
        ],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
    {
        "id": "AIACT_ART5_005",
        "regulation": "eu_ai_act",
        "article": "Article 5",
        "source_paragraph": "5(1)(e)",
        "title": "Restricted use — emotion recognition in workplaces and educational institutions",
        "actor": ["provider", "deployer"],
        "requirement_type": "conditional",
        "requirement": (
            "The use of AI systems for emotion recognition in workplaces and "
            "educational institutions is prohibited except where strictly necessary "
            "for medical or safety reasons. Where an exception applies, documented "
            "medical or safety justification is required."
        ),
        "evidence_required": [
            "risk register",
            "transparency notice",
            "fundamental rights impact assessment",
        ],
        "severity": "critical",
        "linked_articles": [],
        "penalty_relevance": True,
    },
]

# ── GDPR obligations — for evidence_mapper.py ────────────────────────────────
# Actor mapping: GDPR "controller" → "deployer", GDPR "processor" → "provider"
# Source: GDPR (Regulation (EU) 2016/679) articles read from data/regulatory/gdpr/
# Evidence terms: keys from evidence_mapper._EVIDENCE_SYNONYMS only.
_GDPR_OBLIGATIONS: list[dict] = [

    # ── Article 5 — Principles relating to processing of personal data ────────
    {
        "id": "GDPR_ART5_001",
        "regulation": "gdpr",
        "article": "Article 5",
        "source_paragraph": "5(1)(a)-(f)",
        "title": "Data processing principles — lawfulness, fairness, transparency, purpose limitation, minimisation, accuracy, storage limitation, integrity",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "Personal data shall be: processed lawfully, fairly and in a transparent "
            "manner; collected for specified, explicit and legitimate purposes and not "
            "further processed incompatibly; adequate, relevant and limited to what is "
            "necessary; accurate and kept up to date; kept no longer than necessary; "
            "processed with appropriate security. The controller shall be responsible "
            "for and able to demonstrate compliance with these principles."
        ),
        "evidence_required": ["data governance documentation", "transparency notice"],
        "severity": "critical",
        "linked_articles": ["Article 6", "Article 24"],
        "penalty_relevance": True,
    },

    # ── Article 6 — Lawfulness of processing ─────────────────────────────────
    {
        "id": "GDPR_ART6_001",
        "regulation": "gdpr",
        "article": "Article 6",
        "source_paragraph": "6(1)",
        "title": "Legal basis for processing personal data",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "Processing shall be lawful only if at least one of the following applies: "
            "the data subject has given consent; processing is necessary for performance "
            "of a contract; processing is necessary for compliance with a legal "
            "obligation; processing is necessary to protect vital interests; processing "
            "is necessary for a task in the public interest; or processing is necessary "
            "for the legitimate interests of the controller or a third party."
        ),
        "evidence_required": ["data governance documentation", "transparency notice"],
        "severity": "critical",
        "linked_articles": ["Article 5"],
        "penalty_relevance": True,
    },

    # ── Article 9 — Special categories of personal data ───────────────────────
    {
        "id": "GDPR_ART9_001",
        "regulation": "gdpr",
        "article": "Article 9",
        "source_paragraph": "9(1)-(2)",
        "title": "Special categories — prohibition and conditions for processing",
        "actor": ["deployer"],
        "requirement_type": "conditional",
        "requirement": (
            "Processing of personal data revealing racial or ethnic origin, political "
            "opinions, religious beliefs, trade union membership, genetic data, biometric "
            "data for unique identification, health data, or data concerning sex life or "
            "sexual orientation shall be prohibited unless one of the Article 9(2) "
            "exceptions applies. Where an exception applies, explicit consent or a "
            "documented legal basis with appropriate safeguards must be in place."
        ),
        "evidence_required": [
            "data governance documentation",
            "fundamental rights impact assessment",
        ],
        "severity": "critical",
        "linked_articles": ["Article 6", "Article 35"],
        "penalty_relevance": True,
    },

    # ── Article 13 — Information to be provided to data subjects ──────────────
    {
        "id": "GDPR_ART13_001",
        "regulation": "gdpr",
        "article": "Article 13",
        "source_paragraph": "13(1)-(2)",
        "title": "Transparency — information to data subjects at time of data collection",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "Where personal data are collected from the data subject, the controller "
            "shall at the time of collection provide: identity and contact details of "
            "the controller; purposes and legal basis for processing; recipients of "
            "data; storage period; data subject rights (access, rectification, erasure, "
            "restriction, portability, objection); right to withdraw consent; right to "
            "lodge a complaint; and existence of automated decision-making including "
            "profiling with meaningful information about the logic involved."
        ),
        "evidence_required": ["transparency notice"],
        "severity": "high",
        "linked_articles": ["Article 6", "Article 22"],
        "penalty_relevance": True,
    },

    # ── Article 22 — Automated individual decision-making including profiling ─
    {
        "id": "GDPR_ART22_001",
        "regulation": "gdpr",
        "article": "Article 22",
        "source_paragraph": "22(1)-(3)",
        "title": "Automated decision-making — right not to be subject to solely automated decisions",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "The data subject shall have the right not to be subject to a decision "
            "based solely on automated processing, including profiling, which produces "
            "legal effects or similarly significantly affects them. Where an exception "
            "under Article 22(2) applies, the controller shall implement suitable "
            "measures to safeguard data subject rights, at least the right to obtain "
            "human intervention, to express their point of view, and to contest the "
            "decision."
        ),
        "evidence_required": [
            "human oversight procedure",
            "explainability mechanism",
            "appeals mechanism",
        ],
        "severity": "critical",
        "linked_articles": ["Article 13", "Article 9"],
        "penalty_relevance": True,
    },

    # ── Article 24 — Responsibility of the controller ─────────────────────────
    {
        "id": "GDPR_ART24_001",
        "regulation": "gdpr",
        "article": "Article 24",
        "source_paragraph": "24(1)",
        "title": "Controller accountability — technical and organisational measures",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "The controller shall implement appropriate technical and organisational "
            "measures to ensure and to be able to demonstrate that processing is "
            "performed in accordance with the GDPR. Those measures shall be reviewed "
            "and updated where necessary."
        ),
        "evidence_required": ["data governance documentation"],
        "severity": "high",
        "linked_articles": ["Article 5", "Article 25"],
        "penalty_relevance": True,
    },

    # ── Article 25 — Data protection by design and by default ─────────────────
    {
        "id": "GDPR_ART25_001",
        "regulation": "gdpr",
        "article": "Article 25",
        "source_paragraph": "25(1)-(2)",
        "title": "Privacy by design and by default",
        "actor": ["provider", "deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "The controller shall implement appropriate technical and organisational "
            "measures designed to implement data protection principles effectively and "
            "to integrate the necessary safeguards into the processing. By default, "
            "only personal data which are necessary for each specific purpose shall "
            "be processed."
        ),
        "evidence_required": ["data governance documentation"],
        "severity": "high",
        "linked_articles": ["Article 5", "Article 24"],
        "penalty_relevance": True,
    },

    # ── Article 30 — Records of processing activities ─────────────────────────
    {
        "id": "GDPR_ART30_001",
        "regulation": "gdpr",
        "article": "Article 30",
        "source_paragraph": "30(1)",
        "title": "Records of processing activities — controller obligation",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "Each controller shall maintain a record of processing activities under "
            "its responsibility containing: the purposes of processing; categories of "
            "data subjects and personal data; categories of recipients; envisaged time "
            "limits for erasure; and a description of technical and organisational "
            "security measures."
        ),
        "evidence_required": ["logging and audit trail", "data governance documentation"],
        "severity": "high",
        "linked_articles": ["Article 5", "Article 32"],
        "penalty_relevance": True,
    },

    # ── Article 32 — Security of processing ───────────────────────────────────
    {
        "id": "GDPR_ART32_001",
        "regulation": "gdpr",
        "article": "Article 32",
        "source_paragraph": "32(1)-(2)",
        "title": "Security of processing — technical and organisational measures",
        "actor": ["provider", "deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "The controller and the processor shall implement appropriate technical "
            "and organisational measures to ensure a level of security appropriate to "
            "the risk, including as appropriate: pseudonymisation and encryption of "
            "personal data; ability to ensure ongoing confidentiality, integrity, "
            "availability and resilience of processing systems; ability to restore "
            "availability and access in a timely manner after an incident; a process "
            "for regularly testing and evaluating the effectiveness of security "
            "measures."
        ),
        "evidence_required": ["mitigation controls", "safety validation records"],
        "severity": "high",
        "linked_articles": ["Article 5", "Article 30"],
        "penalty_relevance": True,
    },

    # ── Article 35 — Data protection impact assessment ────────────────────────
    {
        "id": "GDPR_ART35_001",
        "regulation": "gdpr",
        "article": "Article 35",
        "source_paragraph": "35(1)-(3)",
        "title": "Data protection impact assessment — required for high-risk processing",
        "actor": ["deployer"],
        "requirement_type": "mandatory",
        "requirement": (
            "Where processing is likely to result in a high risk to the rights and "
            "freedoms of natural persons, the controller shall carry out a data "
            "protection impact assessment. The assessment shall include: a description "
            "of the processing operations and purposes; an assessment of the necessity "
            "and proportionality of the processing; an assessment of risks to the "
            "rights and freedoms of data subjects; and measures envisaged to address "
            "the risks."
        ),
        "evidence_required": ["fundamental rights impact assessment"],
        "severity": "critical",
        "linked_articles": ["Article 9", "Article 22"],
        "penalty_relevance": True,
    },
]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _repo_root = Path(__file__).parent.parent
    print(f"  [ok]  wrote {len(records)} obligations -> {path.relative_to(_repo_root)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write deterministic EU AI Act obligation JSONL files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip if present).",
    )
    args = parser.parse_args()

    targets = [
        (OUT_DIR / "article_6_annex_iii.jsonl",         _ANNEX_III_OBLIGATIONS,    "Annex III (applicability_engine)"),
        (OUT_DIR / "articles_9_15.jsonl",               _ARTICLES_9_15_OBLIGATIONS, "Articles 9–15 (evidence_mapper)"),
        (OUT_DIR / "article_5.jsonl",                   _ARTICLE_5_OBLIGATIONS,    "Article 5 (evidence_mapper)"),
        (GDPR_OUT_DIR / "gdpr_obligations.jsonl",       _GDPR_OBLIGATIONS,         "GDPR Articles (evidence_mapper)"),
    ]

    skipped = 0
    for path, records, label in targets:
        if path.exists() and not args.overwrite:
            print(f"  [--]  {path.name} already exists — skipping ({label}). Use --overwrite to regenerate.")
            skipped += 1
            continue
        _write_jsonl(path, records)

    if skipped:
        print(f"\n  {skipped} file(s) skipped. Pass --overwrite to force regeneration.")

    # Validate: no Article 0, no unknown actors, all evidence terms non-empty
    _validate()
    return 0


def _validate() -> None:
    valid_actors = {"provider", "deployer", "importer", "distributor"}
    errors: list[str] = []
    all_files = list(OUT_DIR.glob("*.jsonl")) + list(GDPR_OUT_DIR.glob("*.jsonl"))

    for path in all_files:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                # Article 0 check
                art = rec.get("article", "")
                if "Article 0" in art or art == "Article 0":
                    errors.append(f"{path.name}:{i} — 'Article 0' detected (id={rec.get('id')})")

                # Actor check (skip Annex III file which has no actor field)
                for actor in rec.get("actor", []):
                    if actor not in valid_actors:
                        errors.append(
                            f"{path.name}:{i} — unknown actor '{actor}' "
                            f"(id={rec.get('id')}). Must be one of {valid_actors}"
                        )

                # Evidence terms non-empty
                for term in rec.get("evidence_required", []):
                    if not term.strip():
                        errors.append(f"{path.name}:{i} — empty evidence_required term (id={rec.get('id')})")

    if errors:
        print("\n  [VALIDATION FAILED]")
        for e in errors:
            print(f"    ✗ {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n  [validation] all {len(all_files)} file(s) passed — no Article 0, no unknown actors.")


if __name__ == "__main__":
    sys.exit(main())
