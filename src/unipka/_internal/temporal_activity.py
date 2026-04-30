"""Optional Temporal activity heartbeats — no-op outside an activity context."""

from __future__ import annotations

import logging

from temporalio import activity

_logger = logging.getLogger("unimol_free_energy.inference")


def heartbeat(*details: object) -> None:
    """Send a Temporal activity heartbeat when in an activity context; no-op otherwise."""
    if not activity.in_activity():
        return
    if details:
        payload = details[0] if len(details) == 1 else details
        _logger.debug("temporal heartbeat: %s", payload)
    activity.heartbeat(*details)
