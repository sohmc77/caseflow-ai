from django.conf import settings
from django.db import models

from agents.schemas import ActionKind

from .checklists import CASE_TYPE_CHOICES


class Case(models.Model):
    class Status(models.TextChoices):
        OPEN = "open"
        READY_FOR_SUBMISSION = "ready_for_submission"
        CLOSED = "closed"

    title = models.CharField(max_length=200)
    case_type = models.CharField(max_length=40, choices=CASE_TYPE_CHOICES)
    applicant_name = models.CharField(max_length=200)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.OPEN)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"#{self.pk} {self.title}"


class CaseDocument(models.Model):
    class Status(models.TextChoices):
        PENDING_REVIEW = "pending_review"
        ACCEPTED = "accepted"
        REJECTED = "rejected"

    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name="documents")
    file = models.FileField(upload_to="documents/%Y/%m/")
    original_filename = models.CharField(max_length=255)
    text = models.TextField(help_text="Extracted plain text; the demo accepts UTF-8 text files only.")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING_REVIEW)
    document_type = models.CharField(max_length=40, blank=True)
    # Only set when a human approves an accept_document action: a validated
    # DocumentExtractionResult, never raw model output.
    extraction = models.JSONField(null=True, blank=True)
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="+")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["uploaded_at", "id"]


class RequirementChecklistItem(models.Model):
    class Status(models.TextChoices):
        OPEN = "open"
        SATISFIED = "satisfied"

    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name="checklist")
    code = models.SlugField(max_length=60)
    title = models.CharField(max_length=200)
    accepted_document_types = models.JSONField(default=list)
    rules = models.JSONField(default=list)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.OPEN)
    satisfied_by = models.ForeignKey(CaseDocument, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")

    class Meta:
        ordering = ["id"]
        constraints = [models.UniqueConstraint(fields=["case", "code"], name="unique_requirement_per_case")]


class AgentRun(models.Model):
    """One execution of the review pipeline. Holds the validated result and full trace."""

    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        SUCCEEDED = "succeeded"
        FAILED = "failed"

    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name="agent_runs")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    provider = models.CharField(max_length=40)
    triggered_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="+")
    result = models.JSONField(null=True, blank=True)  # CaseReviewResult minus traces
    traces = models.JSONField(default=list)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            # At most one active review per case, enforced by the database.
            models.UniqueConstraint(
                fields=["case"],
                condition=models.Q(status__in=["queued", "running"]),
                name="one_active_run_per_case",
            )
        ]


class ProposedAction(models.Model):
    """An AI suggestion. It has no effect until a human approves it."""

    class Status(models.TextChoices):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"
        SUPERSEDED = "superseded"  # a newer review replaced it before anyone decided

    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name="proposed_actions")
    agent_run = models.ForeignKey(AgentRun, on_delete=models.CASCADE, related_name="proposed_actions")
    kind = models.CharField(max_length=40, choices=[(k.value, k.value) for k in ActionKind])
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    target_document = models.ForeignKey(
        CaseDocument, null=True, blank=True, on_delete=models.CASCADE, related_name="proposed_actions"
    )
    target_requirement = models.ForeignKey(
        RequirementChecklistItem, null=True, blank=True, on_delete=models.CASCADE, related_name="+"
    )
    rationale = models.TextField()
    payload = models.JSONField(default=dict, blank=True)
    decided_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT, related_name="+"
    )
    decided_at = models.DateTimeField(null=True, blank=True)
    decision_note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [models.Index(fields=["case", "status"])]


class AuditLogEntry(models.Model):
    """Append-only record of who (human or agent) did what."""

    class ActorType(models.TextChoices):
        USER = "user"
        AGENT = "agent"
        SYSTEM = "system"

    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name="audit_log")
    actor_type = models.CharField(max_length=10, choices=ActorType.choices)
    actor_user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT, related_name="+"
    )
    actor_label = models.CharField(max_length=100, blank=True)
    event = models.CharField(max_length=60)
    target_type = models.CharField(max_length=40, blank=True)
    target_id = models.PositiveBigIntegerField(null=True, blank=True)
    data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
        verbose_name_plural = "audit log entries"

    def save(self, *args, **kwargs):
        if self.pk is not None:
            raise ValueError("audit log entries are append-only")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ValueError("audit log entries are append-only")
