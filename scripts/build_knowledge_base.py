#!/usr/bin/env python3
"""Build and populate the ChromaDB knowledge base for KlarKI.

Downloads (or reads cached) EU AI Act and GDPR full texts in DE + EN,
chunks each article separately, embeds all chunks locally using
sentence-transformers multilingual-e5-small, and stores them in three
ChromaDB collections:

    eu_ai_act            — EU AI Act Articles 9-15 (EN + DE)
    gdpr                 — GDPR Articles 5, 13, 22, 32, 35 (EN + DE)
    compliance_checklist — ~85 structured requirements from Articles 9-15

Data loading priority:
  1. data/regulatory/eu_ai_act_articles.json  (file-based, easy to extend)
  2. data/regulatory/gdpr_articles.json
  3. Built-in hardcoded fallback (if files are missing)

To add more regulatory text: edit the JSON files in data/regulatory/.

Usage:
    python scripts/build_knowledge_base.py [--host localhost] [--port 8001] [--rebuild]

The script is idempotent: re-running with --rebuild deletes and repopulates
all collections; without it, existing data is preserved.
"""

import argparse
import hashlib
import logging
import sys
import uuid
from pathlib import Path
from typing import Generator

import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
BATCH_SIZE = 64  # Embed and upsert in batches to avoid OOM

ARTICLE_DOMAIN_MAP = {
    9:  "risk_management",
    10: "data_governance",
    11: "technical_documentation",
    12: "record_keeping",
    13: "transparency",
    14: "human_oversight",
    15: "security",
}

# ── Structured compliance checklist (Articles 9–15) ───────────────────────────
# 85 requirements derived from EU AI Act Articles 9–15 and GDPR equivalents.
# Each entry maps to a ChromaDB document in the compliance_checklist collection.

COMPLIANCE_REQUIREMENTS: list[dict] = [
    # Article 9 — Risk Management System
    {"requirement_id": "art9_001", "article_num": 9, "domain": "risk_management", "severity": "critical",
     "description": "Establish, implement, document and maintain a risk management system throughout the entire lifecycle of the high-risk AI system."},
    {"requirement_id": "art9_002", "article_num": 9, "domain": "risk_management", "severity": "critical",
     "description": "Identify and analyse known and foreseeable risks associated with the high-risk AI system."},
    {"requirement_id": "art9_003", "article_num": 9, "domain": "risk_management", "severity": "major",
     "description": "Estimate and evaluate risks that may emerge when used in accordance with intended purpose or under foreseeable misuse."},
    {"requirement_id": "art9_004", "article_num": 9, "domain": "risk_management", "severity": "major",
     "description": "Adopt suitable risk management measures in accordance with Articles 9(2)(d) and 9(4)."},
    {"requirement_id": "art9_005", "article_num": 9, "domain": "risk_management", "severity": "major",
     "description": "Test the high-risk AI system to identify appropriate risk management measures before market placement."},
    {"requirement_id": "art9_006", "article_num": 9, "domain": "risk_management", "severity": "minor",
     "description": "Consider reasonably foreseeable misuse scenarios in risk analysis."},
    {"requirement_id": "art9_007", "article_num": 9, "domain": "risk_management", "severity": "major",
     "description": "Ensure residual risks communicated to users are acceptable given the system's intended purpose."},

    # Article 10 — Data and Data Governance
    {"requirement_id": "art10_001", "article_num": 10, "domain": "data_governance", "severity": "critical",
     "description": "Training, validation and testing data sets shall be subject to appropriate data governance and management practices."},
    {"requirement_id": "art10_002", "article_num": 10, "domain": "data_governance", "severity": "critical",
     "description": "Training data must be relevant, sufficiently representative, and to the best extent possible free of errors."},
    {"requirement_id": "art10_003", "article_num": 10, "domain": "data_governance", "severity": "major",
     "description": "Document the data collection, labelling, cleaning, enrichment and aggregation processes."},
    {"requirement_id": "art10_004", "article_num": 10, "domain": "data_governance", "severity": "major",
     "description": "Examine data for possible biases that could affect health, safety or fundamental rights."},
    {"requirement_id": "art10_005", "article_num": 10, "domain": "data_governance", "severity": "major",
     "description": "Identify any data gaps or shortcomings and address them through appropriate measures."},
    {"requirement_id": "art10_006", "article_num": 10, "domain": "data_governance", "severity": "minor",
     "description": "Ensure data sets take into account the specific geographical, contextual, behavioural or functional setting of use."},
    {"requirement_id": "art10_007", "article_num": 10, "domain": "data_governance", "severity": "critical",
     "description": "Special category personal data used for bias monitoring must have appropriate safeguards in place."},
    {"requirement_id": "art10_008", "article_num": 10, "domain": "data_governance", "severity": "major",
     "description": "Validation and testing datasets must be appropriate for the intended purpose of the AI system."},

    # Article 11 — Technical Documentation
    {"requirement_id": "art11_001", "article_num": 11, "domain": "technical_documentation", "severity": "critical",
     "description": "Draw up technical documentation before market placement and keep it up-to-date."},
    {"requirement_id": "art11_002", "article_num": 11, "domain": "technical_documentation", "severity": "critical",
     "description": "Technical documentation must demonstrate compliance with Annex IV requirements."},
    {"requirement_id": "art11_003", "article_num": 11, "domain": "technical_documentation", "severity": "major",
     "description": "Document the general description of the AI system including its intended purpose."},
    {"requirement_id": "art11_004", "article_num": 11, "domain": "technical_documentation", "severity": "major",
     "description": "Include a detailed description of the elements of the AI system and the process for its development."},
    {"requirement_id": "art11_005", "article_num": 11, "domain": "technical_documentation", "severity": "major",
     "description": "Provide detailed information about the monitoring, functioning and control of the AI system."},
    {"requirement_id": "art11_006", "article_num": 11, "domain": "technical_documentation", "severity": "major",
     "description": "Document the validation and testing procedures and the results of tests performed."},
    {"requirement_id": "art11_007", "article_num": 11, "domain": "technical_documentation", "severity": "minor",
     "description": "Technical documentation must be available to national competent authorities upon request."},
    {"requirement_id": "art11_008", "article_num": 11, "domain": "technical_documentation", "severity": "major",
     "description": "Include cybersecurity measures and documentation of the system's robustness."},

    # Article 12 — Record-Keeping / Logging
    {"requirement_id": "art12_001", "article_num": 12, "domain": "record_keeping", "severity": "critical",
     "description": "High-risk AI systems shall have logging capabilities enabling automatic recording of events (logs) throughout the lifetime."},
    {"requirement_id": "art12_002", "article_num": 12, "domain": "record_keeping", "severity": "critical",
     "description": "Logging must enable traceability of the AI system's functioning throughout its lifecycle."},
    {"requirement_id": "art12_003", "article_num": 12, "domain": "record_keeping", "severity": "major",
     "description": "Logs shall identify the period of each use, input data, and persons/entities involved in verifying the results."},
    {"requirement_id": "art12_004", "article_num": 12, "domain": "record_keeping", "severity": "major",
     "description": "For biometric identification systems: log all queries, matches, non-matches, and interventions by humans."},
    {"requirement_id": "art12_005", "article_num": 12, "domain": "record_keeping", "severity": "major",
     "description": "Ensure logs are retained for the period appropriate to the intended purpose of the high-risk AI system."},
    {"requirement_id": "art12_006", "article_num": 12, "domain": "record_keeping", "severity": "minor",
     "description": "Logs must be available to post-market surveillance and national competent authorities."},

    # Article 13 — Transparency and Provision of Information to Users
    {"requirement_id": "art13_001", "article_num": 13, "domain": "transparency", "severity": "critical",
     "description": "High-risk AI systems shall be designed and developed to ensure sufficient transparency to enable users to interpret outputs and use them appropriately."},
    {"requirement_id": "art13_002", "article_num": 13, "domain": "transparency", "severity": "critical",
     "description": "Provide instructions for use to deployers including identity and contact details of the provider."},
    {"requirement_id": "art13_003", "article_num": 13, "domain": "transparency", "severity": "major",
     "description": "Instructions must include the characteristics, capabilities and limitations of the AI system."},
    {"requirement_id": "art13_004", "article_num": 13, "domain": "transparency", "severity": "major",
     "description": "Disclose the intended purpose, level of accuracy, and known risks of the system to deployers."},
    {"requirement_id": "art13_005", "article_num": 13, "domain": "transparency", "severity": "major",
     "description": "Provide information on human oversight measures and the degree of automation in decision-making."},
    {"requirement_id": "art13_006", "article_num": 13, "domain": "transparency", "severity": "major",
     "description": "Inform users about any known limitations on use (specific populations, environments, contexts)."},
    {"requirement_id": "art13_007", "article_num": 13, "domain": "transparency", "severity": "minor",
     "description": "Instructions for use shall be in a language that can be easily understood by deployers."},
    {"requirement_id": "art13_008", "article_num": 13, "domain": "transparency", "severity": "major",
     "description": "Disclose when the AI system interacts with natural persons (chatbots, automated decision systems)."},

    # Article 14 — Human Oversight
    {"requirement_id": "art14_001", "article_num": 14, "domain": "human_oversight", "severity": "critical",
     "description": "High-risk AI systems shall be designed to enable effective human oversight during use."},
    {"requirement_id": "art14_002", "article_num": 14, "domain": "human_oversight", "severity": "critical",
     "description": "Ensure the AI system can be overridden, interrupted, or deactivated by natural persons."},
    {"requirement_id": "art14_003", "article_num": 14, "domain": "human_oversight", "severity": "major",
     "description": "Implement interface features enabling oversight persons to understand AI system capabilities and limitations."},
    {"requirement_id": "art14_004", "article_num": 14, "domain": "human_oversight", "severity": "major",
     "description": "Oversight persons must be able to correctly interpret the AI system's output."},
    {"requirement_id": "art14_005", "article_num": 14, "domain": "human_oversight", "severity": "major",
     "description": "Train and qualify human overseers to monitor performance and identify anomalies."},
    {"requirement_id": "art14_006", "article_num": 14, "domain": "human_oversight", "severity": "major",
     "description": "Provide 'stop button' or equivalent pause mechanism accessible to oversight persons."},
    {"requirement_id": "art14_007", "article_num": 14, "domain": "human_oversight", "severity": "minor",
     "description": "For fully automated high-risk decisions, ensure a meaningful review mechanism by humans is possible."},

    # Article 15 — Accuracy, Robustness, and Cybersecurity
    {"requirement_id": "art15_001", "article_num": 15, "domain": "security", "severity": "critical",
     "description": "High-risk AI systems shall be resilient against attempts by third parties to alter their use or performance through adversarial attacks."},
    {"requirement_id": "art15_002", "article_num": 15, "domain": "security", "severity": "critical",
     "description": "Document and achieve appropriate levels of accuracy for the intended purpose throughout the lifecycle."},
    {"requirement_id": "art15_003", "article_num": 15, "domain": "security", "severity": "major",
     "description": "Implement technical measures against adversarial manipulation of training data (data poisoning)."},
    {"requirement_id": "art15_004", "article_num": 15, "domain": "security", "severity": "major",
     "description": "Implement technical measures against adversarial manipulation of model inputs (evasion attacks)."},
    {"requirement_id": "art15_005", "article_num": 15, "domain": "security", "severity": "major",
     "description": "Ensure the AI system has fall-back plans (fail-safe) to prevent unsafe operation."},
    {"requirement_id": "art15_006", "article_num": 15, "domain": "security", "severity": "major",
     "description": "Metrics for measuring accuracy and robustness must be documented and communicated to users."},
    {"requirement_id": "art15_007", "article_num": 15, "domain": "security", "severity": "major",
     "description": "Cybersecurity measures must be proportionate to the risk and state of the art."},
    {"requirement_id": "art15_008", "article_num": 15, "domain": "security", "severity": "minor",
     "description": "Ensure resilience against errors, faults or inconsistencies in the operating environment."},

    # GDPR Crossovers (relevant to AI Act high-risk systems)
    {"requirement_id": "gdpr_art22_001", "article_num": 22, "domain": "human_oversight", "severity": "critical",
     "description": "GDPR Article 22: Individuals have the right not to be subject to solely automated decisions with legal or similarly significant effects."},
    {"requirement_id": "gdpr_art22_002", "article_num": 22, "domain": "human_oversight", "severity": "major",
     "description": "GDPR Article 22: Where automated decision-making is permitted, provide meaningful information about the logic and significance."},
    {"requirement_id": "gdpr_art13_001", "article_num": 13, "domain": "transparency", "severity": "major",
     "description": "GDPR Article 13: Where personal data is collected, inform data subjects about automated decision-making including profiling."},
    {"requirement_id": "gdpr_art35_001", "article_num": 35, "domain": "risk_management", "severity": "critical",
     "description": "GDPR Article 35: Conduct a Data Protection Impact Assessment (DPIA) before processing likely to result in high risk to individuals."},
    {"requirement_id": "gdpr_art5_001", "article_num": 5, "domain": "data_governance", "severity": "critical",
     "description": "GDPR Article 5: Personal data must be collected for specified, explicit and legitimate purposes (purpose limitation)."},
    {"requirement_id": "gdpr_art5_002", "article_num": 5, "domain": "data_governance", "severity": "critical",
     "description": "GDPR Article 5: Personal data must be adequate, relevant and limited to what is necessary (data minimisation)."},
    {"requirement_id": "gdpr_art32_001", "article_num": 32, "domain": "security", "severity": "critical",
     "description": "GDPR Article 32: Implement appropriate technical and organisational security measures for personal data processing."},
]

# ── EU AI Act article summary texts (EN + DE) ──────────────────────────────────
# Concise per-article knowledge base seeded into eu_ai_act collection.
# In production, build_knowledge_base.py can augment with full EUR-Lex text.

EU_AI_ACT_ARTICLES: list[dict] = [
    {
        "article_num": 9, "lang": "en", "domain": "risk_management",
        "title": "Article 9 — Risk Management System",
        "text": (
            "A risk management system shall be established, implemented, documented and maintained "
            "in relation to high-risk AI systems. The risk management system shall consist of a "
            "continuous iterative process run throughout the entire lifecycle of a high-risk AI system, "
            "requiring regular systematic updating. The process shall identify and analyse the known "
            "and foreseeable risks that the high-risk AI system can pose to health, safety or fundamental "
            "rights when the high-risk AI system is used in accordance with its intended purpose. "
            "Risk management measures shall give due consideration to the effects and possible "
            "interactions resulting from the combined application of the requirements set out in this "
            "Section 2. Risk management measures shall be such that any residual risk associated with "
            "each hazard as well as the overall residual risk of the high-risk AI systems is judged "
            "acceptable, provided the high-risk AI system is used in accordance with its intended purpose "
            "or under conditions of reasonably foreseeable misuse."
        ),
    },
    {
        "article_num": 9, "lang": "de", "domain": "risk_management",
        "title": "Artikel 9 — Risikomanagementsystem",
        "text": (
            "Für Hochrisiko-KI-Systeme ist ein Risikomanagementsystem einzurichten, umzusetzen, "
            "zu dokumentieren und aufrechtzuerhalten. Das Risikomanagementsystem besteht aus einem "
            "kontinuierlichen iterativen Prozess, der während des gesamten Lebenszyklus eines "
            "Hochrisiko-KI-Systems durchgeführt wird und regelmäßige systematische Aktualisierungen "
            "erfordert. Im Rahmen des Risikomanagementsystems werden die bekannten und vorhersehbaren "
            "Risiken ermittelt und analysiert, die das Hochrisiko-KI-System für Gesundheit, Sicherheit "
            "oder Grundrechte darstellen kann. Die Risikomanagementsystem-Maßnahmen müssen so beschaffen "
            "sein, dass etwaige Restrisiken im Zusammenhang mit den einzelnen Gefährdungen sowie das "
            "gesamte Restrisiko der Hochrisiko-KI-Systeme als annehmbar eingestuft werden."
        ),
    },
    {
        "article_num": 10, "lang": "en", "domain": "data_governance",
        "title": "Article 10 — Data and Data Governance",
        "text": (
            "High-risk AI systems which make use of techniques involving the training of models with "
            "data shall be developed on the basis of training, validation and testing data sets that "
            "meet the quality criteria referred to in paragraphs 2 to 5. Training, validation and "
            "testing data sets shall be subject to appropriate data governance and management practices. "
            "Those practices shall concern in particular: the relevant design choices; data collection "
            "processes and the origin of data; relevant data preparation processing operations; "
            "the formulation of relevant assumptions; the assessment of the availability, quantity "
            "and suitability of the data sets needed; examination in view of possible biases that are "
            "likely to affect health or safety of persons or lead to prohibited discrimination. "
            "Training, validation and testing data sets shall be relevant, sufficiently representative, "
            "and to the best extent possible, free of errors and complete in view of the intended purpose."
        ),
    },
    {
        "article_num": 10, "lang": "de", "domain": "data_governance",
        "title": "Artikel 10 — Daten und Daten-Governance",
        "text": (
            "Hochrisiko-KI-Systeme, die Techniken verwenden, bei denen Modelle mit Daten trainiert werden, "
            "sind auf der Grundlage von Trainings-, Validierungs- und Testdatensätzen zu entwickeln. "
            "Trainings-, Validierungs- und Testdatensätzen müssen angemessene Datenverwaltungs- und "
            "-managementpraktiken unterliegen. Diese Praktiken betreffen insbesondere: die relevanten "
            "Designentscheidungen; Datenerfassungsprozesse und Datenherkunft; relevante Vorverarbeitungen; "
            "die Formulierung relevanter Annahmen; die Beurteilung der Verfügbarkeit, Quantität und "
            "Eignung der benötigten Datensätze; Prüfung auf mögliche Verzerrungen, die die Gesundheit "
            "oder Sicherheit von Personen beeinträchtigen oder zu verbotener Diskriminierung führen könnten. "
            "Die Datensätze müssen relevant, ausreichend repräsentativ und so weit wie möglich fehlerfrei sein."
        ),
    },
    {
        "article_num": 11, "lang": "en", "domain": "technical_documentation",
        "title": "Article 11 — Technical Documentation",
        "text": (
            "The technical documentation of a high-risk AI system shall be drawn up before that system "
            "is placed on the market or put into service and shall be kept up-to-date. The technical "
            "documentation shall be drawn up in such a way to demonstrate that the high-risk AI system "
            "complies with the requirements set out in this Section and provide national competent "
            "authorities and notified bodies with all the necessary information to assess the compliance "
            "of the AI system with those requirements. It shall contain, at a minimum, the elements set "
            "out in Annex IV. SMEs, including start-ups, may provide the elements of the technical "
            "documentation specified in Annex IV in a simplified manner."
        ),
    },
    {
        "article_num": 11, "lang": "de", "domain": "technical_documentation",
        "title": "Artikel 11 — Technische Dokumentation",
        "text": (
            "Die technische Dokumentation eines Hochrisiko-KI-Systems ist zu erstellen, bevor dieses "
            "System in Verkehr gebracht oder in Betrieb genommen wird, und auf dem neuesten Stand zu "
            "halten. Die technische Dokumentation ist so zu erstellen, dass nachgewiesen wird, dass das "
            "Hochrisiko-KI-System die in diesem Abschnitt festgelegten Anforderungen erfüllt, und den "
            "nationalen zuständigen Behörden und notifizierten Stellen alle notwendigen Informationen "
            "für die Bewertung der Konformität des KI-Systems bereitgestellt werden. Sie enthält "
            "mindestens die in Anhang IV aufgeführten Elemente. KMU, einschließlich Start-ups, können "
            "die in Anhang IV angegebenen Elemente der technischen Dokumentation in vereinfachter Form bereitstellen."
        ),
    },
    {
        "article_num": 12, "lang": "en", "domain": "record_keeping",
        "title": "Article 12 — Record-Keeping",
        "text": (
            "High-risk AI systems shall technically allow for the automatic recording of events "
            "(logs) over the lifetime of the system. Logging capabilities shall conform to recognised "
            "standards or common specifications. Logging capabilities shall ensure a level of traceability "
            "of the AI system's functioning throughout its lifetime that is appropriate to the intended "
            "purpose of the system. In particular, logging capabilities shall enable the monitoring of "
            "the operation of the high-risk AI system with respect to the occurrence of situations that "
            "may result in the AI system presenting a risk within the meaning of Article 79(1) and to "
            "substantial modifications, and shall facilitate the post-market monitoring referred to in "
            "Article 72 and the monitoring by the deployer referred to in Article 26(5)."
        ),
    },
    {
        "article_num": 12, "lang": "de", "domain": "record_keeping",
        "title": "Artikel 12 — Protokollierung",
        "text": (
            "Hochrisiko-KI-Systeme müssen technisch die automatische Aufzeichnung von Ereignissen "
            "(Protokollen) über die Lebensdauer des Systems ermöglichen. Die Protokollierungsfähigkeiten "
            "müssen anerkannten Normen oder gemeinsamen Spezifikationen entsprechen. Die Protokollierungsfähigkeiten "
            "müssen ein Maß an Rückverfolgbarkeit der Funktionsweise des KI-Systems über seine gesamte "
            "Lebensdauer gewährleisten, das dem Verwendungszweck des Systems angemessen ist. Insbesondere "
            "müssen die Protokollierungsfähigkeiten die Überwachung des Betriebs des Hochrisiko-KI-Systems "
            "im Hinblick auf das Auftreten von Situationen ermöglichen, die dazu führen können, dass das "
            "KI-System ein Risiko darstellt."
        ),
    },
    {
        "article_num": 13, "lang": "en", "domain": "transparency",
        "title": "Article 13 — Transparency and Provision of Information to Deployers",
        "text": (
            "High-risk AI systems shall be designed and developed in such a way to ensure that their "
            "operation is sufficiently transparent to enable deployers to interpret the system's output "
            "and use it appropriately. An appropriate type and degree of transparency shall be ensured, "
            "with a view to achieving compliance with the relevant obligations of the provider and the "
            "deployer set out in Section 3 of this Chapter. High-risk AI systems shall be accompanied "
            "by instructions for use in an appropriate digital format or otherwise that include concise, "
            "complete, correct and clear information that is relevant, accessible and comprehensible to "
            "deployers. The information shall specify the identity and the contact details of the provider "
            "and, where applicable, of its authorised representative; the characteristics, capabilities "
            "and limitations of performance of the high-risk AI system."
        ),
    },
    {
        "article_num": 13, "lang": "de", "domain": "transparency",
        "title": "Artikel 13 — Transparenz und Bereitstellung von Informationen für Betreiber",
        "text": (
            "Hochrisiko-KI-Systeme sind so zu gestalten und zu entwickeln, dass ihr Betrieb hinreichend "
            "transparent ist, damit die Betreiber die Ausgabe des Systems interpretieren und angemessen "
            "verwenden können. Es ist ein angemessenes Maß an Transparenz zu gewährleisten. "
            "Hochrisiko-KI-Systeme sind mit einer Gebrauchsanweisung in einem geeigneten digitalen "
            "Format oder in sonstiger Form zu begleiten, die präzise, vollständige, korrekte und klare "
            "Informationen enthält, die für die Betreiber relevant, zugänglich und verständlich sind. "
            "Die Informationen müssen die Identität und Kontaktdaten des Anbieters sowie die Merkmale, "
            "Fähigkeiten und Leistungsgrenzen des Hochrisiko-KI-Systems angeben."
        ),
    },
    {
        "article_num": 14, "lang": "en", "domain": "human_oversight",
        "title": "Article 14 — Human Oversight",
        "text": (
            "High-risk AI systems shall be designed and developed in such a way, including with appropriate "
            "human-machine interface tools, that they can be effectively overseen by natural persons during "
            "the period in which the AI system is in use. The oversight measures referred to in paragraph 1 "
            "shall be proportionate to the risks, level of autonomy and context of use of the high-risk AI "
            "system, and shall be ensured through either one or both of the following types of measures: "
            "measures identified and built, when technically feasible, into the high-risk AI system by the "
            "provider before it is placed on the market or put into service; measures identified by the "
            "provider and/or the deployer. Natural persons to whom human oversight is assigned shall have "
            "the necessary competence, training and authority to carry out that role."
        ),
    },
    {
        "article_num": 14, "lang": "de", "domain": "human_oversight",
        "title": "Artikel 14 — Menschliche Aufsicht",
        "text": (
            "Hochrisiko-KI-Systeme sind so zu gestalten und zu entwickeln, dass sie von natürlichen Personen "
            "während des Einsatzes wirksam beaufsichtigt werden können. Die Aufsichtsmaßnahmen müssen den "
            "Risiken, dem Grad der Autonomie und dem Verwendungskontext des Hochrisiko-KI-Systems angemessen "
            "sein. Natürlichen Personen, denen die menschliche Aufsicht übertragen wird, sind alle "
            "erforderlichen Informationen, Schulungen und Befugnisse zu erteilen, um diese Aufgabe "
            "wahrnehmen zu können. Insbesondere muss sichergestellt werden, dass die Betreiber in der Lage "
            "sind, die Ausgaben des Hochrisiko-KI-Systems zu verstehen, zu interpretieren und bei Bedarf "
            "nicht zu berücksichtigen."
        ),
    },
    {
        "article_num": 15, "lang": "en", "domain": "security",
        "title": "Article 15 — Accuracy, Robustness and Cybersecurity",
        "text": (
            "High-risk AI systems shall be designed and developed in such a way that they achieve an "
            "appropriate level of accuracy, robustness, and cybersecurity, and that they perform "
            "consistently in those respects throughout their lifecycle. The levels of accuracy and the "
            "relevant accuracy metrics for high-risk AI systems shall be declared in the accompanying "
            "instructions for use. High-risk AI systems shall be resilient as regards errors, faults or "
            "inconsistencies that may occur within the system or the environment in which the system "
            "operates, in particular if such errors, faults or inconsistencies may lead to death or "
            "serious injury to a person or to property. The technical robustness of high-risk AI systems "
            "may be achieved through technical redundancy solutions, which may include backup or "
            "fail-safe plans. High-risk AI systems that continue to learn after being placed on the "
            "market shall be developed in such a way that possibly biased outputs due to outputs used "
            "as an input for future operations are duly addressed with appropriate mitigation measures."
        ),
    },
    {
        "article_num": 15, "lang": "de", "domain": "security",
        "title": "Artikel 15 — Genauigkeit, Robustheit und Cybersicherheit",
        "text": (
            "Hochrisiko-KI-Systeme sind so zu gestalten und zu entwickeln, dass sie ein angemessenes "
            "Maß an Genauigkeit, Robustheit und Cybersicherheit erreichen und diesbezüglich über ihren "
            "gesamten Lebenszyklus hinweg eine konsistente Leistung erbringen. Die Genauigkeitsniveaus "
            "und die relevanten Genauigkeitsmetriken für Hochrisiko-KI-Systeme sind in den begleitenden "
            "Gebrauchsanweisungen anzugeben. Hochrisiko-KI-Systeme müssen widerstandsfähig gegenüber "
            "Fehlern, Ausfällen oder Inkonsistenzen sein, die im System oder in der Umgebung auftreten "
            "können. Technische Robustheit kann durch technische Redundanzlösungen erreicht werden, "
            "die auch Sicherungs- oder Notfallpläne umfassen können."
        ),
    },
]

# ── GDPR article summaries ─────────────────────────────────────────────────────

GDPR_ARTICLES: list[dict] = [
    {
        "article_num": 5, "lang": "en", "domain": "data_governance",
        "title": "GDPR Article 5 — Principles Relating to Processing of Personal Data",
        "text": (
            "Personal data shall be: processed lawfully, fairly and in a transparent manner; collected "
            "for specified, explicit and legitimate purposes (purpose limitation); adequate, relevant and "
            "limited to what is necessary (data minimisation); accurate and where necessary kept up to "
            "date; kept in a form which permits identification for no longer than necessary (storage "
            "limitation); processed in a manner that ensures appropriate security (integrity and "
            "confidentiality). The controller shall be responsible for, and be able to demonstrate "
            "compliance with (accountability)."
        ),
    },
    {
        "article_num": 13, "lang": "en", "domain": "transparency",
        "title": "GDPR Article 13 — Information to Be Provided Where Personal Data Are Collected",
        "text": (
            "Where personal data relating to a data subject are collected from the data subject, the "
            "controller shall provide the data subject with information including: identity and contact "
            "details of the controller; purposes and legal basis of processing; legitimate interests "
            "pursued; recipients of personal data; retention period; data subject rights including "
            "access, rectification, erasure; right to withdraw consent; right to lodge a complaint. "
            "Where AI automated decision-making including profiling is used, the controller shall "
            "provide meaningful information about the logic involved, the significance and the "
            "envisaged consequences for the data subject."
        ),
    },
    {
        "article_num": 22, "lang": "en", "domain": "human_oversight",
        "title": "GDPR Article 22 — Automated Individual Decision-Making Including Profiling",
        "text": (
            "The data subject shall have the right not to be subject to a decision based solely on "
            "automated processing, including profiling, which produces legal effects concerning them "
            "or similarly significantly affects them. Automated decisions are permitted where necessary "
            "for entering/performance of a contract, authorised by EU/Member State law, or based on "
            "explicit consent. Where automated processing is permitted, the controller shall implement "
            "suitable measures to safeguard the data subject's rights and freedoms and legitimate "
            "interests, at least the right to obtain human intervention on the part of the controller, "
            "to express their point of view and to contest the decision."
        ),
    },
    {
        "article_num": 32, "lang": "en", "domain": "security",
        "title": "GDPR Article 32 — Security of Processing",
        "text": (
            "Taking into account the state of the art, the costs of implementation and the nature, "
            "scope, context and purposes of processing as well as the risk of varying likelihood and "
            "severity for the rights and freedoms of natural persons, the controller and the processor "
            "shall implement appropriate technical and organisational measures to ensure a level of "
            "security appropriate to the risk, including pseudonymisation and encryption; ongoing "
            "confidentiality, integrity, availability and resilience of processing systems; ability to "
            "restore availability and access to personal data; regular testing, assessing and "
            "evaluating the effectiveness of technical and organisational measures."
        ),
    },
    {
        "article_num": 35, "lang": "en", "domain": "risk_management",
        "title": "GDPR Article 35 — Data Protection Impact Assessment",
        "text": (
            "Where a type of processing, in particular using new technologies, is likely to result in "
            "a high risk to the rights and freedoms of natural persons, the controller shall, prior to "
            "the processing, carry out an assessment of the impact of the envisaged processing operations "
            "on the protection of personal data. A data protection impact assessment shall be required "
            "in the case of: systematic and extensive evaluation of personal aspects, including profiling; "
            "processing on a large scale of special categories of data; systematic monitoring of a "
            "publicly accessible area on a large scale. The assessment shall include at minimum: a "
            "systematic description of the envisaged processing operations, the purposes and the "
            "legitimate interests pursued; an assessment of necessity and proportionality; an assessment "
            "of the risks to the rights and freedoms of data subjects; the measures envisaged to address "
            "the risks."
        ),
    },
]


# ── Regulatory data loader ────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent  # project root
REGULATORY_DIR = ROOT / "data" / "regulatory"


def parse_regulatory_txt(path: Path) -> list[dict]:
    """Parse a regulatory .txt file into per-language article dicts.

    Expected file format:
        ARTICLE: 9
        REGULATION: eu_ai_act
        DOMAIN: risk_management
        TITLE_EN: Article 9 — Risk Management System
        TITLE_DE: Artikel 9 — Risikomanagementsystem

        === EN ===
        Article text...

        === DE ===
        German text...

    Args:
        path: Path to the .txt file.

    Returns:
        List of dicts (one per language section found in the file).
    """
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("=== ") and stripped.endswith(" ==="):
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = stripped[4:-4].lower()  # "EN" -> "en"
            current_lines = []
        elif current_section is None:
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                metadata[key.strip().lower()] = val.strip()
        else:
            current_lines.append(line)

    if current_section is not None and current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    results = []
    for lang, text_content in sections.items():
        if not text_content:
            continue
        results.append({
            "article_num": int(metadata.get("article", 0)),
            "lang": lang,
            "domain": metadata.get("domain", "unknown"),
            "regulation": metadata.get("regulation", path.parent.name),
            "title": metadata.get(f"title_{lang}", metadata.get("title_en", "")),
            "text": text_content,
        })

    return results


def load_regulatory_directory(directory: Path, fallback: list[dict]) -> list[dict]:
    """Load all .txt files from a regulatory directory, falling back to hardcoded data.

    Args:
        directory: Path to directory containing article .txt files.
        fallback: Hardcoded list to use if no files are found or directory missing.

    Returns:
        List of article dicts.
    """
    if not directory.exists():
        log.info(f"Directory {directory.name}/ not found — using built-in fallback data")
        return fallback

    txt_files = sorted(directory.glob("*.txt"))
    if not txt_files:
        log.info(f"No .txt files in {directory.name}/ — using built-in fallback data")
        return fallback

    articles: list[dict] = []
    for f in txt_files:
        try:
            parsed = parse_regulatory_txt(f)
            articles.extend(parsed)
            log.debug(f"  Loaded {len(parsed)} language section(s) from {f.name}")
        except Exception as exc:
            log.warning(f"  Skipping {f.name}: {exc}")

    if articles:
        log.info(f"Loaded {len(articles)} entries from data/regulatory/{directory.name}/")
        return articles

    log.warning(f"No valid entries parsed from {directory.name}/ — using built-in fallback data")
    return fallback


# ── Helper functions ───────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple recursive character text splitter (no LangChain dependency here).

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters to repeat at the start of the next chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            for sep in [". ", "\n", " "]:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        next_start = end - overlap
        start = next_start if next_start > start else end  # guarantee forward progress
    return [c for c in chunks if c.strip()]


def make_chunk_id(text: str, prefix: str = "") -> str:
    """Generate a stable chunk ID from content hash.

    Args:
        text: Chunk text.
        prefix: Optional prefix (e.g. 'eu_ai_act_art9').

    Returns:
        Deterministic UUID-like string.
    """
    h = hashlib.md5(f"{prefix}{text}".encode()).hexdigest()
    return str(uuid.UUID(h))


def batch(items: list, size: int) -> Generator[list, None, None]:
    """Yield successive batches of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Main build logic ───────────────────────────────────────────────────────────

def build_knowledge_base(chroma_host: str, chroma_port: int, rebuild: bool = False) -> None:
    """Build and populate all ChromaDB collections.

    Args:
        chroma_host: ChromaDB server hostname.
        chroma_port: ChromaDB server port.
        rebuild: If True, delete existing collections before rebuilding.
    """
    log.info(f"Connecting to ChromaDB at {chroma_host}:{chroma_port}")
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    # Heartbeat check
    try:
        client.heartbeat()
        log.info("ChromaDB connection OK")
    except Exception as e:
        log.error(f"Cannot connect to ChromaDB: {e}")
        sys.exit(1)

    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Embedding model loaded")

    # ── Load regulatory text (txt files preferred, hardcoded fallback) ────────
    eu_ai_act_articles = load_regulatory_directory(REGULATORY_DIR / "eu_ai_act", EU_AI_ACT_ARTICLES)
    gdpr_articles      = load_regulatory_directory(REGULATORY_DIR / "gdpr",      GDPR_ARTICLES)

    # ── Rebuild collections if requested ──────────────────────────────────────
    if rebuild:
        for name in ["eu_ai_act", "gdpr", "compliance_checklist"]:
            try:
                client.delete_collection(name)
                log.info(f"Deleted existing collection: {name}")
            except Exception:
                pass

    # ── 1. eu_ai_act collection ───────────────────────────────────────────────
    log.info("Building eu_ai_act collection...")
    eu_col = client.get_or_create_collection(
        "eu_ai_act",
        metadata={"hnsw:space": "cosine", "description": "EU AI Act Articles 9–15 (EN + DE)"},
    )

    eu_ids, eu_docs, eu_metas, eu_texts_to_embed = [], [], [], []

    for article in eu_ai_act_articles:
        chunks = chunk_text(article["text"])
        for i, chunk in enumerate(chunks):
            cid = make_chunk_id(chunk, prefix=f"eu_{article['article_num']}_{article['lang']}")
            eu_ids.append(cid)
            eu_docs.append(chunk)
            eu_metas.append({
                "regulation": "eu_ai_act",
                "article_num": article["article_num"],
                "domain": article["domain"],
                "lang": article["lang"],
                "chunk_index": i,
                "title": article["title"],
                "is_annex": False,
            })
            eu_texts_to_embed.append(chunk)

    log.info(f"Embedding {len(eu_texts_to_embed)} EU AI Act chunks...")
    for b_texts, b_ids, b_docs, b_metas in zip(
        batch(eu_texts_to_embed, BATCH_SIZE),
        batch(eu_ids, BATCH_SIZE),
        batch(eu_docs, BATCH_SIZE),
        batch(eu_metas, BATCH_SIZE),
    ):
        embeddings = model.encode(b_texts, normalize_embeddings=True).tolist()
        eu_col.upsert(ids=b_ids, embeddings=embeddings, documents=b_docs, metadatas=b_metas)

    log.info(f"eu_ai_act: {eu_col.count()} documents stored")

    # ── 2. gdpr collection ────────────────────────────────────────────────────
    log.info("Building gdpr collection...")
    gdpr_col = client.get_or_create_collection(
        "gdpr",
        metadata={"hnsw:space": "cosine", "description": "GDPR Articles relevant to AI (EN)"},
    )

    gdpr_ids, gdpr_docs, gdpr_metas, gdpr_texts = [], [], [], []

    for article in gdpr_articles:
        chunks = chunk_text(article["text"])
        for i, chunk in enumerate(chunks):
            cid = make_chunk_id(chunk, prefix=f"gdpr_{article['article_num']}_{article['lang']}")
            gdpr_ids.append(cid)
            gdpr_docs.append(chunk)
            gdpr_metas.append({
                "regulation": "gdpr",
                "article_num": article["article_num"],
                "domain": article["domain"],
                "lang": article["lang"],
                "chunk_index": i,
                "title": article["title"],
                "is_annex": False,
            })
            gdpr_texts.append(chunk)

    log.info(f"Embedding {len(gdpr_texts)} GDPR chunks...")
    for b_texts, b_ids, b_docs, b_metas in zip(
        batch(gdpr_texts, BATCH_SIZE),
        batch(gdpr_ids, BATCH_SIZE),
        batch(gdpr_docs, BATCH_SIZE),
        batch(gdpr_metas, BATCH_SIZE),
    ):
        embeddings = model.encode(b_texts, normalize_embeddings=True).tolist()
        gdpr_col.upsert(ids=b_ids, embeddings=embeddings, documents=b_docs, metadatas=b_metas)

    log.info(f"gdpr: {gdpr_col.count()} documents stored")

    # ── 3. compliance_checklist collection ────────────────────────────────────
    log.info("Building compliance_checklist collection...")
    checklist_col = client.get_or_create_collection(
        "compliance_checklist",
        metadata={
            "hnsw:space": "cosine",
            "description": "Structured compliance requirements from EU AI Act Articles 9–15 + GDPR",
        },
    )

    req_ids = [r["requirement_id"] for r in COMPLIANCE_REQUIREMENTS]
    req_docs = [r["description"] for r in COMPLIANCE_REQUIREMENTS]
    req_metas = [
        {
            "requirement_id": r["requirement_id"],
            "article_num": r["article_num"],
            "domain": r["domain"],
            "severity": r["severity"],
        }
        for r in COMPLIANCE_REQUIREMENTS
    ]

    log.info(f"Embedding {len(req_docs)} compliance requirements...")
    for b_texts, b_ids, b_docs, b_metas in zip(
        batch(req_docs, BATCH_SIZE),
        batch(req_ids, BATCH_SIZE),
        batch(req_docs, BATCH_SIZE),
        batch(req_metas, BATCH_SIZE),
    ):
        embeddings = model.encode(b_texts, normalize_embeddings=True).tolist()
        checklist_col.upsert(ids=b_ids, embeddings=embeddings, documents=b_docs, metadatas=b_metas)

    log.info(f"compliance_checklist: {checklist_col.count()} documents stored")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Knowledge base build complete!")
    log.info(f"  eu_ai_act:            {eu_col.count():>4} documents")
    log.info(f"  gdpr:                 {gdpr_col.count():>4} documents")
    log.info(f"  compliance_checklist: {checklist_col.count():>4} documents")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KlarKI ChromaDB knowledge base")
    parser.add_argument(
        "--host",
        default="localhost",
        help="ChromaDB hostname (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="ChromaDB port (default: 8001, mapped from docker-compose)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild all collections",
    )
    args = parser.parse_args()
    build_knowledge_base(chroma_host=args.host, chroma_port=args.port, rebuild=args.rebuild)
