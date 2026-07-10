"""MiniMax image generation backend.

Exposes MiniMax's image-01 model as an ImageGenProvider implementation.

- image-01: text-to-image + image-to-image (subject_reference)
  - supports: text, image_url, reference_image_urls (1 cap)

API reference: https://platform.minimax.io/docs/api-reference/image-generation-t2i

Env / base URL resolution (CN-first, global fallback):
  1. ``MINIMAX_CN_API_KEY`` + base ``https://api.minimaxi.com`` (CN endpoint)
  2. ``MINIMAX_API_KEY``   + base ``https://api.minimax.io``  (global endpoint)
  3. ``MINIMAX_BASE_URL`` / ``MINIMAX_CN_BASE_URL`` env overrides accepted,
     but the path is always ``/v1/image_generation`` so the two hosts are
     interchangeable for this endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

CN_HOST = "https://api.minimaxi.com"
GLOBAL_HOST = "https://api.minimax.io"
DEFAULT_BASE = "/v1/image_generation"

DEFAULT_MODEL = "image-01"

# image-01 native aspect ratios per docs.
_ASPECT_TO_RATIO = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

_MODELS: dict[str, dict[str, Any]] = {
    "image-01": {
        "display": "MiniMax image-01",
        "speed": "~5-15s",
        "strengths": "Text-to-image + image-to-image with subject_reference",
        "price": "paid",
    },
}


# ---------------------------------------------------------------------------
# Base URL / key resolution
# ---------------------------------------------------------------------------


def _resolve_endpoint() -> dict[str, str]:
    """Return ``{"base_url": ..., "api_key": ..., "region": "cn"|"global"}``.

    CN takes priority when its key is set — matches the user's primary
    configuration. Falls back to global. Raises-style via empty api_key if
    neither is configured (caller turns this into auth_required error).
    """
    cn_key = os.environ.get("MINIMAX_CN_API_KEY", "").strip()
    if cn_key:
        base = os.environ.get("MINIMAX_CN_BASE_URL", "").strip().rstrip("/")
        # If MINIMAX_CN_BASE_URL already ends with the path, use as-is;
        # otherwise append the canonical path.
        if base.endswith("/v1/image_generation") or base.endswith("/v1"):
            url = base if base.endswith("/image_generation") else f"{base}/image_generation"
        else:
            url = f"{CN_HOST}{DEFAULT_BASE}"
        return {"base_url": url, "api_key": cn_key, "region": "cn"}

    gl_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if gl_key:
        base = os.environ.get("MINIMAX_BASE_URL", "").strip().rstrip("/")
        if base.endswith("/v1/image_generation") or base.endswith("/v1"):
            url = base if base.endswith("/image_generation") else f"{base}/image_generation"
        else:
            url = f"{GLOBAL_HOST}{DEFAULT_BASE}"
        return {"base_url": url, "api_key": gl_key, "region": "global"}

    return {"base_url": f"{GLOBAL_HOST}{DEFAULT_BASE}", "api_key": "", "region": "global"}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _post_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 120,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            detail_json = json.loads(detail)
            detail = detail_json.get("base_resp", {}).get("status_msg") or detail
        except Exception:
            pass
        return {
            "__error__": True,
            "status": exc.code,
            "detail": (detail or "")[:500],
        }
    except Exception as exc:
        return {"__error__": True, "status": 0, "detail": f"{exc}"[:500]}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax (image-01) image generation backend."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax (海螺)"

    def is_available(self) -> bool:
        ep = _resolve_endpoint()
        return bool(ep.get("api_key"))

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta.get("price", "paid"),
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> str | None:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "MiniMax (海螺)",
            "badge": "paid",
            "tag": "image-01 — T2I + I2I with subject_reference (CN endpoint by default)",
            "env_vars": [
                {
                    "key": "MINIMAX_CN_API_KEY",
                    "prompt": "MiniMax CN API key (api.minimaxi.com)",
                    "url": "https://api.minimaxi.com/user-center/basic-information/interface-key",
                },
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax global API key (api.minimax.io) — fallback",
                    "url": "https://www.minimax.io/user-center/basic-information/interface-key",
                },
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 1}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="minimax",
                aspect_ratio=aspect,
            )

        ep = _resolve_endpoint()
        api_key = ep["api_key"]
        base_url = ep["base_url"]
        region = ep["region"]

        if not api_key:
            return error_response(
                error=(
                    "No MiniMax API key configured. Set MINIMAX_CN_API_KEY "
                    "(api.minimaxi.com) or MINIMAX_API_KEY (api.minimax.io)."
                ),
                error_type="auth_required",
                provider="minimax",
                aspect_ratio=aspect,
            )

        # image-01 is the only model right now; ignore the configured pick.
        model_id = DEFAULT_MODEL

        # Collect subject reference: image_url wins, then reference_image_urls.
        sources: list[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            if ref and ref not in sources:
                sources.append(ref)
        sources = sources[:1]  # image-01 only accepts 1 subject_reference
        is_edit = bool(sources)
        modality = "image" if is_edit else "text"

        payload: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt[:2000],
            "aspect_ratio": _ASPECT_TO_RATIO.get(aspect, "16:9"),
            "response_format": "url",
            "n": 1,
        }
        if is_edit:
            payload["subject_reference"] = [
                {"type": "character", "image_file": sources[0]}
            ]
        # prompt_optimizer is on by default — make it opt-out via kwargs.
        if "prompt_optimizer" in kwargs:
            payload["prompt_optimizer"] = bool(kwargs["prompt_optimizer"])
        else:
            payload["prompt_optimizer"] = True
        if "seed" in kwargs and kwargs["seed"] is not None:
            payload["seed"] = int(kwargs["seed"])

        resp = _post_json(base_url, api_key, payload)
        if resp.get("__error__"):
            return error_response(
                error=f"MiniMax image API error {resp['status']}: {resp['detail']}",
                error_type="api_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # image-01 returns: {"data": {"image_urls": [...]}, ...}
        data = resp.get("data") or {}
        image_urls: list[str] = list(data.get("image_urls") or [])
        image_b64: list[str] = list(data.get("image_base64") or [])

        if not image_urls and not image_b64:
            base_resp = resp.get("base_resp") or {}
            msg = base_resp.get("status_msg") or "MiniMax returned no image data"
            return error_response(
                error=f"{msg} (raw: {json.dumps(resp)[:300]})",
                error_type="empty_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Prefer b64 if present, else download first URL.
        if image_b64:
            try:
                saved_path = save_b64_image(image_b64[0], prefix=f"minimax_{model_id}")
                image_ref = str(saved_path)
            except Exception as exc:
                return error_response(
                    error=f"Could not save MiniMax image: {exc}",
                    error_type="io_error",
                    provider="minimax",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        else:
            try:
                saved_path = save_url_image(image_urls[0], prefix=f"minimax_{model_id}")
                image_ref = str(saved_path)
            except Exception as exc:
                logger.warning("Could not cache MiniMax image URL: %s", exc)
                image_ref = image_urls[0]

        extra: dict[str, Any] = {
            "region": region,
            "aspect_ratio": _ASPECT_TO_RATIO.get(aspect, "16:9"),
        }
        meta = resp.get("metadata") or {}
        if meta.get("success_count") is not None:
            extra["success_count"] = meta["success_count"]
        if resp.get("id"):
            extra["request_id"] = resp["id"]

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="minimax",
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the MiniMax image generation provider."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
