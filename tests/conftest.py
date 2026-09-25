from datetime import date

import pytest

from agents.schemas import (
    CaseSnapshot,
    DocumentExtractionResult,
    DocumentSnapshot,
    ExtractedField,
    RequirementSnapshot,
)
from cases.checklists import TEMPLATES

PASSPORT_TEXT = """REPUBLIC OF ARCADIA
PASSPORT
Surname and given names: OKAFOR, AMARA
Passport No: AR4471902
Nationality: Arcadian
Date of Birth: 02 FEB 1994
Date of Expiry: 14 MAR 2031
"""

BANK_TEXT = """Northbank Savings
Account Statement
Account Holder: Amara Okafor
Statement Date: 2026-08-31
Currency: EUR
Closing Balance: EUR 8,250.00
"""


@pytest.fixture(autouse=True)
def _isolated_settings(settings, tmp_path):
    settings.LLM_PROVIDER = "mock"  # never call a real API from tests
    settings.MEDIA_ROOT = tmp_path / "media"
    settings.CELERY_TASK_ALWAYS_EAGER = True


@pytest.fixture
def snapshot() -> CaseSnapshot:
    return CaseSnapshot(
        id=1,
        case_type="work_permit",
        applicant_name="Amara Okafor",
        as_of=date(2026, 9, 1),
        documents=[
            DocumentSnapshot(id=10, filename="passport.txt", text=PASSPORT_TEXT),
            DocumentSnapshot(id=11, filename="bank.txt", text=BANK_TEXT),
        ],
        requirements=[RequirementSnapshot(**t.model_dump()) for t in TEMPLATES["work_permit"]],
    )


def make_extraction(document_id: int, document_type: str, **fields: str) -> DocumentExtractionResult:
    return DocumentExtractionResult(
        document_id=document_id,
        document_type=document_type,
        fields=[ExtractedField(name=k, value=v, evidence=f"{k}: {v}", confidence=1.0) for k, v in fields.items()],
        confidence=1.0,
    )
