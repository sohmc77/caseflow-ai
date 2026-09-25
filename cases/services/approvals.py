"""The only code path that changes case state because of an AI suggestion.

Approval re-checks each action's preconditions against the *current* database
state, because the case may have changed since the agent looked at it.
"""

from collections.abc import Callable

from django.db import transaction
from django.utils import timezone

from agents.schemas import ActionKind

from ..models import Case, CaseDocument, ProposedAction, RequirementChecklistItem
from . import audit


class ActionConflict(Exception):
    """The action can no longer be applied as proposed (HTTP 409)."""


def approve_action(action_id: int, user, note: str = "") -> ProposedAction:
    with transaction.atomic():
        action = _lock_pending(action_id)
        _APPLY[action.kind](action)
        _decide(action, ProposedAction.Status.APPROVED, user, note)
        audit.record(action.case, "action_approved", user=user, target=action, data={"kind": action.kind})
    return action


def reject_action(action_id: int, user, note: str = "") -> ProposedAction:
    with transaction.atomic():
        action = _lock_pending(action_id)
        _decide(action, ProposedAction.Status.REJECTED, user, note)
        audit.record(action.case, "action_rejected", user=user, target=action, data={"kind": action.kind, "note": note})
    return action


def _lock_pending(action_id: int) -> ProposedAction:
    action = ProposedAction.objects.select_for_update().get(pk=action_id)
    Case.objects.select_for_update().get(pk=action.case_id)  # serialize with reviews/other approvals
    if action.status != ProposedAction.Status.PENDING:
        raise ActionConflict(f"action is already {action.status}")
    return action


def _decide(action: ProposedAction, status: str, user, note: str) -> None:
    action.status = status
    action.decided_by = user
    action.decided_at = timezone.now()
    action.decision_note = note
    action.save(update_fields=["status", "decided_by", "decided_at", "decision_note"])


# --------------------------------------------------------------- effects per kind


def _pending_document(action: ProposedAction) -> CaseDocument:
    doc = CaseDocument.objects.select_for_update().get(pk=action.target_document_id)
    if doc.status != CaseDocument.Status.PENDING_REVIEW:
        raise ActionConflict(f"document {doc.pk} is already {doc.status}")
    return doc


def _accept_document(action: ProposedAction) -> None:
    doc = _pending_document(action)
    doc.status = CaseDocument.Status.ACCEPTED
    doc.extraction = action.payload["extraction"]
    doc.document_type = doc.extraction["document_type"]
    doc.save(update_fields=["status", "extraction", "document_type"])


def _reject_document(action: ProposedAction) -> None:
    doc = _pending_document(action)
    doc.status = CaseDocument.Status.REJECTED
    doc.save(update_fields=["status"])


def _mark_requirement_satisfied(action: ProposedAction) -> None:
    requirement = RequirementChecklistItem.objects.select_for_update().get(pk=action.target_requirement_id)
    if requirement.status == RequirementChecklistItem.Status.SATISFIED:
        raise ActionConflict(f"requirement '{requirement.code}' is already satisfied")
    doc = action.target_document
    if doc.status != CaseDocument.Status.ACCEPTED:
        raise ActionConflict(f"approve acceptance of document {doc.pk} first (it is {doc.status})")
    requirement.status = RequirementChecklistItem.Status.SATISFIED
    requirement.satisfied_by = doc
    requirement.save(update_fields=["status", "satisfied_by"])


def _change_case_status(action: ProposedAction) -> None:
    case = action.case
    open_items = case.checklist.exclude(status=RequirementChecklistItem.Status.SATISFIED)
    if open_items.exists():
        codes = ", ".join(open_items.values_list("code", flat=True))
        raise ActionConflict(f"requirements still open: {codes}")
    case.status = action.payload["new_status"]
    case.save(update_fields=["status", "updated_at"])


def _no_state_change(action: ProposedAction) -> None:
    """request_document / flag_for_review: approval is the human sign-off itself.

    Delivering the message to the applicant (email, portal) is out of scope for
    this demo. It would be an outbox row written in this same transaction.
    """


_APPLY: dict[str, Callable[[ProposedAction], None]] = {
    ActionKind.ACCEPT_DOCUMENT: _accept_document,
    ActionKind.REJECT_DOCUMENT: _reject_document,
    ActionKind.MARK_REQUIREMENT_SATISFIED: _mark_requirement_satisfied,
    ActionKind.REQUEST_DOCUMENT: _no_state_change,
    ActionKind.FLAG_FOR_REVIEW: _no_state_change,
    ActionKind.CHANGE_CASE_STATUS: _change_case_status,
}
