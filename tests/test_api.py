"""End-to-end through HTTP: upload → review → human approval. Mock provider, eager Celery."""

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

from cases.models import AgentRun, AuditLogEntry, CaseDocument, ProposedAction, RequirementChecklistItem

from .conftest import BANK_TEXT, PASSPORT_TEXT

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    user = get_user_model().objects.create_user("reviewer", password="pw")
    api = APIClient()
    api.force_authenticate(user)
    return api


@pytest.fixture
def case_id(client):
    resp = client.post(
        "/api/cases/", {"title": "Demo", "case_type": "work_permit", "applicant_name": "Amara Okafor"}, format="json"
    )
    assert resp.status_code == 201, resp.data
    return resp.data["id"]


def upload(client, case_id, name, text):
    file = SimpleUploadedFile(name, text.encode(), content_type="text/plain")
    return client.post(f"/api/cases/{case_id}/documents/", {"file": file}, format="multipart")


@pytest.fixture
def reviewed_case(client, case_id, django_capture_on_commit_callbacks):
    upload(client, case_id, "passport.txt", PASSPORT_TEXT)
    upload(client, case_id, "bank.txt", BANK_TEXT)
    with django_capture_on_commit_callbacks(execute=True):
        resp = client.post(f"/api/cases/{case_id}/reviews/")
    assert resp.status_code == 202
    return case_id


def actions(client, case_id, **params):
    return client.get(f"/api/cases/{case_id}/proposed-actions/", params).data["results"]


def find(items, kind, **match):
    return next(a for a in items if a["kind"] == kind and all(a[k] == v for k, v in match.items()))


def test_requires_authentication():
    assert APIClient().get("/api/cases/").status_code == 403


def test_create_case_instantiates_checklist(client, case_id):
    detail = client.get(f"/api/cases/{case_id}/").data
    assert [c["code"] for c in detail["checklist"]] == ["identity", "job_offer", "proof_of_funds"]


def test_upload_rejects_non_text(client, case_id):
    resp = client.post(
        f"/api/cases/{case_id}/documents/",
        {"file": SimpleUploadedFile("scan.pdf", b"%PDF-1.7", content_type="application/pdf")},
        format="multipart",
    )
    assert resp.status_code == 400 and "txt" in str(resp.data["file"])


def test_review_only_proposes_it_never_mutates(client, reviewed_case):
    run = AgentRun.objects.get(case_id=reviewed_case)
    assert run.status == AgentRun.Status.SUCCEEDED, run.error
    assert len(run.traces) == 4

    pending = actions(client, reviewed_case, status="pending")
    assert {a["kind"] for a in pending} >= {"accept_document", "reject_document", "request_document"}
    # The AI proposed accepting and rejecting documents, but nothing has changed yet.
    assert set(CaseDocument.objects.values_list("status", flat=True)) == {"pending_review"}
    assert set(RequirementChecklistItem.objects.values_list("status", flat=True)) == {"open"}


def test_approve_flow_applies_changes_in_order(client, reviewed_case):
    items = actions(client, reviewed_case)
    accept = find(items, "accept_document")
    mark = find(items, "mark_requirement_satisfied", target_requirement="identity")

    # Marking the requirement before the document is accepted is refused.
    resp = client.post(f"/api/proposed-actions/{mark['id']}/approve/")
    assert resp.status_code == 409 and "approve acceptance" in resp.data["detail"]

    resp = client.post(f"/api/proposed-actions/{accept['id']}/approve/", {"note": "checked"}, format="json")
    assert resp.status_code == 200 and resp.data["status"] == "approved"
    doc = CaseDocument.objects.get(pk=accept["target_document"])
    assert doc.status == "accepted" and doc.document_type == "passport"
    assert doc.extraction["fields"]  # the validated extraction travelled with the action

    assert client.post(f"/api/proposed-actions/{mark['id']}/approve/").status_code == 200
    assert RequirementChecklistItem.objects.get(code="identity").status == "satisfied"

    # Deciding twice is a conflict, not a silent no-op.
    assert client.post(f"/api/proposed-actions/{accept['id']}/approve/").status_code == 409

    events = list(AuditLogEntry.objects.filter(case_id=reviewed_case).values_list("event", "actor_type"))
    assert ("review_completed", "agent") in events
    assert events.count(("action_approved", "user")) == 2


def test_reject_leaves_state_untouched(client, reviewed_case):
    reject_doc = find(actions(client, reviewed_case), "reject_document")
    resp = client.post(f"/api/proposed-actions/{reject_doc['id']}/reject/", {"note": "balance is fine"}, format="json")
    assert resp.status_code == 200 and resp.data["status"] == "rejected"
    assert CaseDocument.objects.get(pk=reject_doc["target_document"]).status == "pending_review"


def test_new_review_supersedes_pending_actions(client, reviewed_case, django_capture_on_commit_callbacks):
    first = {a["id"] for a in actions(client, reviewed_case)}
    with django_capture_on_commit_callbacks(execute=True):
        client.post(f"/api/cases/{reviewed_case}/reviews/")
    assert set(ProposedAction.objects.filter(id__in=first).values_list("status", flat=True)) == {"superseded"}
    stale = next(iter(first))
    assert client.post(f"/api/proposed-actions/{stale}/approve/").status_code == 409


def test_only_one_active_review_per_case(client, case_id):
    # Without executing on-commit callbacks the first run stays queued.
    assert client.post(f"/api/cases/{case_id}/reviews/").status_code == 202
    resp = client.post(f"/api/cases/{case_id}/reviews/")
    assert resp.status_code == 409 and resp.data["agent_run"]


def test_provider_failure_is_recorded_not_raised(client, case_id, django_capture_on_commit_callbacks, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("cases.services.review.run_case_review", explode)
    upload(client, case_id, "passport.txt", PASSPORT_TEXT)
    with django_capture_on_commit_callbacks(execute=True):
        run_id = client.post(f"/api/cases/{case_id}/reviews/").data["id"]

    run = client.get(f"/api/agent-runs/{run_id}/").data
    assert run["status"] == "failed" and "provider unavailable" in run["error"]
    assert not ProposedAction.objects.exists()
