"""Guided Annex III risk tier classification wizard.

Processes 9 yes/no questions covering all Annex III categories to produce
a deterministic risk tier for the user's AI system.
"""

import structlog
from models.schemas import RiskTier

logger = structlog.get_logger()

WIZARD_QUESTIONS = [
    {"id": "q1", "text": "Does the system use biometric identification of natural persons?"},
    {"id": "q2", "text": "Is the system used in critical infrastructure (energy, water, transport, finance)?"},
    {"id": "q3", "text": "Is the system used for educational or vocational training assessment?"},
    {"id": "q4", "text": "Is the system used in employment, worker management, or recruitment?"},
    {"id": "q5", "text": "Is the system used for access to essential services (credit, housing, insurance)?"},
    {"id": "q6", "text": "Is the system used in law enforcement or criminal justice?"},
    {"id": "q7", "text": "Is the system used for migration, asylum, or border control decisions?"},
    {"id": "q8", "text": "Is the system used in administration of justice or democratic processes?"},
    {"id": "q9", "text": "Does the system monitor or infer the emotions or behaviour of workers in the workplace?"},
]


async def guided_risk_classification(answers: dict[str, bool]) -> RiskTier:
    """Classify risk tier from questionnaire answers.

    Decision logic (in priority order):
    - Q9 (workplace emotion/behaviour monitoring) → PROHIBITED
    - Any q1–q8 yes                               → HIGH
    - All no                                       → MINIMAL

    Args:
        answers: Dict mapping question IDs ('q1'–'q9') to bool answers.

    Returns:
        RiskTier enum value.
    """
    if answers.get("q9"):
        logger.warning("wizard_prohibited", trigger="q9_workplace_emotion")
        return RiskTier.PROHIBITED

    if any(answers.get(f"q{i}") for i in range(1, 9)):
        triggered = [f"q{i}" for i in range(1, 9) if answers.get(f"q{i}")]
        logger.info("wizard_high_risk", triggers=triggered)
        return RiskTier.HIGH

    logger.info("wizard_minimal_risk")
    return RiskTier.MINIMAL
