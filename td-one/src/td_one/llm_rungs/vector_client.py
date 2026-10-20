"""Vector proxy LLM client for the ladder.

Implements the ``complete(prompt) -> str`` protocol consumed by
``td_one.llm_rungs.runner``, routing through Vector Institute's LLM proxy
to Gemini Flash models. Uses the ``custom.vector-llm-proxy`` credential
via the surrogate exchange; the key never touches this process.

Note: this file was reconstructed 2026-10-20 from the compiled module
after the original (never committed) was lost in a working-tree reset.
Interface and constants are byte-identical to the original bytecode.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request  # noqa: E402

logger = logging.getLogger(__name__)

PROXY_URL = "https://proxy.vectorinstitute.ai/v1/chat/completions"
ALLOWED_HOSTS = ("proxy.vectorinstitute.ai",)
MODELS = {
    "flash-lite": "gemini-3.1-flash-lite-preview",
    "flash": "gemini-3.5-flash",
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a moved endpoint fails loudly, not silently."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN202
        raise RuntimeError(f"redirect refused, verify the endpoint: {newurl}")


class VectorProxyClient:
    """Drop-in replacement for GroqChatClient for the ladder.

    Routes completions through the Vector proxy. ``model`` accepts the
    short names in :data:`MODELS` or a raw model id.
    """

    def __init__(
        self,
        model: str = "flash-lite",
        call_budget: int = 1000,
        max_tokens: int = 2048,
    ) -> None:
        self.model_id = MODELS.get(model, model)
        self.call_budget = call_budget
        self.max_tokens = max_tokens
        self._calls = 0

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        timeout_s: float = 120.0,
    ) -> str:
        """Send one completion, return the text content."""
        if self._calls >= self.call_budget:
            raise RuntimeError(f"call budget exhausted ({self._calls})")
        self._calls += 1
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            PROXY_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        add_surrogate_to_request(req, "custom.vector-llm-proxy", allowed_hosts=ALLOWED_HOSTS)
        opener = urllib.request.build_opener(NoRedirect())
        try:
            with opener.open(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code == 429:
                raise RuntimeError(f"proxy 429 rate limited: {body}") from e
            raise RuntimeError(f"proxy HTTP {e.code}: {body}") from e
        return str(data["choices"][0]["message"]["content"])
