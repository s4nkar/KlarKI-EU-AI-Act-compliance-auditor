"""Annex III risk wizard router.

Endpoints:
    GET  /api/v1/wizard/questions  — Return the 9 yes/no questions
    POST /api/v1/wizard/classify   — Submit answers, get risk tier
"""

from fastapi import APIRouter
from pydantic import BaseModel

from models.schemas import APIResponse, RiskTier
from services.risk_wizard import WIZARD_QUESTIONS, guided_risk_classification

router = APIRouter(prefix="/api/v1/wizard", tags=["wizard"])


class WizardAnswers(BaseModel):
    answers: dict[str, bool]


@router.get("/questions", response_model=APIResponse)
async def get_questions() -> APIResponse:
    """Return the list of Annex III classification questions."""
    return APIResponse(status="success", data={"questions": WIZARD_QUESTIONS})


@router.post("/classify", response_model=APIResponse)
async def classify(body: WizardAnswers) -> APIResponse:
    """Submit yes/no answers and receive a risk tier classification.

    Args:
        body: Dict of question IDs to boolean answers.

    Returns:
        APIResponse with risk_tier string.
    """
    tier: RiskTier = await guided_risk_classification(body.answers)
    return APIResponse(status="success", data={"risk_tier": tier.value})
