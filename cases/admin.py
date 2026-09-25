from django.contrib import admin

from .models import AgentRun, AuditLogEntry, Case, CaseDocument, ProposedAction, RequirementChecklistItem


class ChecklistInline(admin.TabularInline):
    model = RequirementChecklistItem
    extra = 0


@admin.register(Case)
class CaseAdmin(admin.ModelAdmin):
    list_display = ["id", "title", "case_type", "applicant_name", "status", "created_at"]
    list_filter = ["status", "case_type"]
    inlines = [ChecklistInline]


@admin.register(CaseDocument)
class CaseDocumentAdmin(admin.ModelAdmin):
    list_display = ["id", "case", "original_filename", "status", "document_type"]
    list_filter = ["status"]


@admin.register(AgentRun)
class AgentRunAdmin(admin.ModelAdmin):
    list_display = ["id", "case", "status", "provider", "created_at", "finished_at"]
    list_filter = ["status", "provider"]


@admin.register(ProposedAction)
class ProposedActionAdmin(admin.ModelAdmin):
    list_display = ["id", "case", "kind", "status", "decided_by", "decided_at"]
    list_filter = ["status", "kind"]
    # Decisions go through the API/service layer so preconditions and audit apply.
    readonly_fields = [f.name for f in ProposedAction._meta.fields]


@admin.register(AuditLogEntry)
class AuditLogEntryAdmin(admin.ModelAdmin):
    list_display = ["id", "case", "actor_type", "actor_label", "event", "created_at"]
    list_filter = ["actor_type", "event"]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
