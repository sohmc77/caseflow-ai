from django.db import models

from ..models import AuditLogEntry, Case


def record(
    case: Case,
    event: str,
    *,
    user=None,
    agent: str = "",
    target: models.Model | None = None,
    data: dict | None = None,
) -> AuditLogEntry:
    if user is not None:
        actor_type = AuditLogEntry.ActorType.USER
    elif agent:
        actor_type = AuditLogEntry.ActorType.AGENT
    else:
        actor_type = AuditLogEntry.ActorType.SYSTEM
    return AuditLogEntry.objects.create(
        case=case,
        actor_type=actor_type,
        actor_user=user,
        actor_label=agent or (user.get_username() if user else "system"),
        event=event,
        target_type=target._meta.model_name if target is not None else "",
        target_id=target.pk if target is not None else None,
        data=data or {},
    )
