from rest_framework.routers import DefaultRouter

from .views import AgentRunViewSet, CaseViewSet, ProposedActionViewSet

router = DefaultRouter()
router.register("cases", CaseViewSet)
router.register("proposed-actions", ProposedActionViewSet)
router.register("agent-runs", AgentRunViewSet)

urlpatterns = router.urls
