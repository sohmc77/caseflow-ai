from django.core.files.uploadedfile import UploadedFile
from django.db import transaction

from ..checklists import TEMPLATES
from ..models import Case, CaseDocument, RequirementChecklistItem
from . import audit


@transaction.atomic
def create_case(*, title: str, case_type: str, applicant_name: str, user) -> Case:
    case = Case.objects.create(title=title, case_type=case_type, applicant_name=applicant_name, created_by=user)
    RequirementChecklistItem.objects.bulk_create(
        RequirementChecklistItem(
            case=case,
            code=t.code,
            title=t.title,
            accepted_document_types=[d.value for d in t.accepted_document_types],
            rules=[r.model_dump(mode="json") for r in t.rules],
        )
        for t in TEMPLATES[case_type]
    )
    audit.record(case, "case_created", user=user, target=case, data={"case_type": case_type})
    return case


@transaction.atomic
def add_document(*, case: Case, upload: UploadedFile, text: str, user) -> CaseDocument:
    document = CaseDocument.objects.create(
        case=case, file=upload, original_filename=upload.name, text=text, uploaded_by=user
    )
    audit.record(case, "document_uploaded", user=user, target=document, data={"filename": upload.name})
    return document
