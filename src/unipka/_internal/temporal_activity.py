"""Optional Temporal activity heartbeats — no-op outside an activity context."""

from temporalio import activity


def heartbeat() -> None:
    """Send a Temporal activity heartbeat when in an activity context; no-op otherwise."""
    if activity.in_activity():
        activity.heartbeat()
