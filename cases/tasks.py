from celery import shared_task

from .services.review import execute_review


@shared_task(name="cases.run_case_review")
def run_case_review_task(run_id: int) -> None:
    # Failures are recorded on the AgentRun row, not raised. Transient provider
    # errors are already retried inside the provider SDK.
    execute_review(run_id)
