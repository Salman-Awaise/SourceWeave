"""Optional LangSmith tracing.

Tracing is off unless a key is present, regardless of `LANGCHAIN_TRACING_V2`,
and the missing-key warning is emitted at most once per process. Document and
memory content is withheld unless `TRACE_CONTENT=true`.
"""

from __future__ import annotations

import logging
import os

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_warned = False
_configured = False


def configure_tracing(settings: Settings | None = None) -> bool:
    """Set up LangSmith env vars if usable. Returns True when tracing is on."""
    global _warned, _configured
    settings = settings or get_settings()

    if not settings.langchain_tracing_v2:
        _disable()
        return False

    if not settings.langchain_api_key.strip():
        if not _warned:
            logger.warning(
                "LANGCHAIN_TRACING_V2 is on but LANGCHAIN_API_KEY is not set; tracing is disabled."
            )
            _warned = True
        _disable()
        return False

    if not _configured:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        if not settings.trace_content:
            # Keep document text and memories out of the trace payload.
            os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
            os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"
            logger.info(
                "tracing on for project %r with content hidden "
                "(set TRACE_CONTENT=true to include it)",
                settings.langchain_project,
            )
        else:
            logger.warning(
                "TRACE_CONTENT=true: document and memory text will be sent to LangSmith."
            )
        _configured = True
    return True


def _disable() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def reset_tracing_state() -> None:
    """Test helper: allow the warning and configuration to happen again."""
    global _warned, _configured
    _warned = False
    _configured = False
