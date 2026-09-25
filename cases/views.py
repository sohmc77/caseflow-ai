"""Thin HTTP layer: validate input, call a service, serialize the result."""

from django.conf import settings
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response

from .models import AgentRun, Case, ProposedAction
from .serializers import (
    AgentRunSerializer,
    AuditLogEntrySerializer,
    CaseSerializer,
    DecisionSerializer,
    DocumentSerializer,
    DocumentUploadSerializer,
    ProposedActionSerializer,
)
from .services import approvals, cases, review


class CaseViewSet(mixins.ListModelMixin, mixins.CreateModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = Case.objects.prefetch_related("checklist", "documents")
    serializer_class = CaseSerializer

    def perform_create(self, serializer):
        serializer.instance = cases.create_case(**serializer.validated_data, user=self.request.user)

    @action(detail=True, methods=["post"], parser_classes=[MultiPartParser])
    def documents(self, request, pk=None):
        case = self.get_object()
        upload = DocumentUploadSerializer(data=request.data, context={})
        upload.is_valid(raise_exception=True)
        document = cases.add_document(
            case=case, upload=upload.validated_data["file"], text=upload.context["text"], user=request.user
        )
        return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["get", "post"])
    def reviews(self, request, pk=None):
        case = self.get_object()
        if request.method == "GET":
            return self._paginated(case.agent_runs.all(), AgentRunSerializer)
        try:
            run = review.start_review(case, request.user, settings.LLM_PROVIDER)
        except review.ReviewAlreadyRunning as exc:
            return Response(
                {"detail": str(exc), "agent_run": exc.run.pk if exc.run else None},
                status=status.HTTP_409_CONFLICT,
            )
        return Response(AgentRunSerializer(run).data, status=status.HTTP_202_ACCEPTED)

    @action(detail=True, methods=["get"], url_path="proposed-actions")
    def proposed_actions(self, request, pk=None):
        qs = self.get_object().proposed_actions.select_related("target_requirement")
        if wanted := request.query_params.get("status"):
            qs = qs.filter(status=wanted)
        return self._paginated(qs, ProposedActionSerializer)

    @action(detail=True, methods=["get"], url_path="audit-log")
    def audit_log(self, request, pk=None):
        return self._paginated(self.get_object().audit_log.all(), AuditLogEntrySerializer)

    def _paginated(self, qs, serializer_class):
        page = self.paginate_queryset(qs)
        return self.get_paginated_response(serializer_class(page, many=True).data)


class ProposedActionViewSet(mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = ProposedAction.objects.select_related("target_requirement")
    serializer_class = ProposedActionSerializer

    @action(detail=True, methods=["post"])
    def approve(self, request, pk=None):
        return self._decide(approvals.approve_action, request)

    @action(detail=True, methods=["post"])
    def reject(self, request, pk=None):
        return self._decide(approvals.reject_action, request)

    def _decide(self, decide, request):
        action_obj = self.get_object()
        body = DecisionSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        try:
            decided = decide(action_obj.pk, request.user, body.validated_data["note"])
        except approvals.ActionConflict as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        return Response(self.get_serializer(decided).data)


class AgentRunViewSet(mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = AgentRun.objects.all()
    serializer_class = AgentRunSerializer
