from django.conf import settings
from rest_framework import serializers

from .checklists import CASE_TYPE_CHOICES
from .models import AgentRun, AuditLogEntry, Case, CaseDocument, ProposedAction, RequirementChecklistItem


class RequirementSerializer(serializers.ModelSerializer):
    class Meta:
        model = RequirementChecklistItem
        fields = ["id", "code", "title", "accepted_document_types", "rules", "status", "satisfied_by"]


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = CaseDocument
        fields = ["id", "original_filename", "status", "document_type", "extraction", "uploaded_at"]


class CaseSerializer(serializers.ModelSerializer):
    case_type = serializers.ChoiceField(choices=CASE_TYPE_CHOICES)
    checklist = RequirementSerializer(many=True, read_only=True)
    documents = DocumentSerializer(many=True, read_only=True)

    class Meta:
        model = Case
        fields = ["id", "title", "case_type", "applicant_name", "status", "checklist", "documents", "created_at"]
        read_only_fields = ["status"]


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

    def validate_file(self, upload):
        if upload.size > settings.MAX_UPLOAD_BYTES:
            raise serializers.ValidationError("file too large (max 2 MB)")
        if not upload.name.lower().endswith(".txt"):
            raise serializers.ValidationError("this demo accepts .txt documents only (no OCR)")
        try:
            self.context["text"] = upload.read().decode("utf-8")
        except UnicodeDecodeError:
            raise serializers.ValidationError("file must be UTF-8 text") from None
        upload.seek(0)
        return upload


class AgentRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentRun
        fields = [
            "id",
            "case",
            "status",
            "provider",
            "result",
            "traces",
            "error",
            "created_at",
            "started_at",
            "finished_at",
        ]


class ProposedActionSerializer(serializers.ModelSerializer):
    target_requirement = serializers.SlugRelatedField(slug_field="code", read_only=True)

    class Meta:
        model = ProposedAction
        fields = [
            "id",
            "case",
            "agent_run",
            "kind",
            "status",
            "target_document",
            "target_requirement",
            "rationale",
            "payload",
            "decided_by",
            "decided_at",
            "decision_note",
            "created_at",
        ]


class DecisionSerializer(serializers.Serializer):
    note = serializers.CharField(required=False, allow_blank=True, max_length=2000, default="")


class AuditLogEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditLogEntry
        fields = ["id", "actor_type", "actor_label", "event", "target_type", "target_id", "data", "created_at"]
