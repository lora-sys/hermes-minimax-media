"""MiniMax video generation backend.

Exposes MiniMax's video models as a VideoGenProvider implementation:

- MiniMax-Hailuo-2.3: T2V + I2V, 6/10s, 720p/1080p
- MiniMax-Hailuo-02:  T2V + I2V, 6/10s, 720p/1080p (start-end frames supported)
- S2V-01: subject-reference T2V with face image, 6s, 1080p

API reference: https://platform.minimax.io/docs/api-reference/video-generation-t2v

Workflow (async, polled):
    1. POST /v1/video_generation       -> task_id
    2. GET  /v1/query/video_generation  (10s cadence) -> file_id when success
    3. GET  /v1/files/retrieve?file_id= -> download_url
    4. GET <download_url>              -> bytes -> cache

Env / base URL resolution (CN-first, global fallback):
    1. ``MINIMAX_CN_API_KEY`` + host ``https://api.minimaxi.com``  (CN)
    2. ``MINIMAX_API_KEY``   + host ``https://api.minimax.io``    (global)
    3. ``MINIMAX_CN_BASE_URL`` / ``MINIMAX_BASE_URL`` may override the host;
       the three endpoints all share the same path layout, so the swap is safe.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    save_bytes_video,
    success_response,
)

logger = logging.getLogger(__name__)

CN_HOST = "https://api.minimaxi.com"
GLOBAL_HOST = "https://api.minimax.io"

VIDEO_GEN_PATH = "/v1/video_generation"
QUERY_PATH = "/v1/query/video_generation"
FILES_PATH = "/v1/files/retrieve"

DEFAULT_MODEL = "MiniMax-Hailuo-2.3"

# Per the docs: poll every 10s. Total budget: ~5 minutes.
POLL_INTERVAL_SEC = 10
POLL_TIMEOUT_SEC = 300  # 5 min

# Model catalog. Each model lists what it supports so we can route correctly.
_MODELS: dict[str, dict[str, Any]] = {
    "MiniMax-Hailuo-2.3": {
        "display": "Hailuo 2.3",
        "speed": "~30-60s for 6s clip",
        "strengths": "Latest flagship T2V/I2V — best quality",
        "price": "paid",
        "durations": [6, 10],
        "resolutions": ["768P", "1080P"],
        "supports_i2v": True,
        "supports_subject_ref": False,
    },
    "MiniMax-Hailuo-02": {
        "display": "Hailuo 02",
        "speed": "~30-60s for 6s clip",
        "strengths": "Start-end frame support, 1080p, fast",
        "price": "paid",
        "durations": [6, 10],
        "resolutions": ["768P", "1080P"],
        "supports_i2v": True,
        "supports_subject_ref": False,
    },
    "S2V-01": {
        "display": "S2V-01 (subject reference)",
        "speed": "~60-90s for 6s clip",
        "strengths": "Face-consistent subject reference video",
        "price": "paid",
        "durations": [6],
        "resolutions": ["1080P"],
        "supports_i2v": False,
        "supports_subject_ref": True,
    },
}

# aspect_ratio string passthrough — provider doesn't need to remap.
_VALID_ASPECTS = {"16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"}


# ---------------------------------------------------------------------------
# Endpoint / key resolution
# ---------------------------------------------------------------------------


def _resolve_endpoint() -> dict[str, str]:
    """Return ``{"host": ..., "api_key": ..., "region": "cn"|"global"}``.

    CN takes priority when its key is set. ``MINIMAX_CN_BASE_URL`` /
    ``MINIMAX_BASE_URL`` override the host (we still strip any path prefix to
    keep the well-known ``/v1/...`` paths correct).
    """
    cn_key = os.environ.get("MINIMAX_CN_API_KEY", "").strip()
    if cn_key:
        host = _host_from_env("MINIMAX_CN_BASE_URL", CN_HOST)
        return {"host": host, "api_key": cn_key, "region": "cn"}

    gl_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if gl_key:
        host = _host_from_env("MINIMAX_BASE_URL", GLOBAL_HOST)
        return {"host": host, "api_key": gl_key, "region": "global"}

    return {"host": GLOBAL_HOST, "api_key": "", "region": "global"}


def _host_from_env(env_var: str, default_host: str) -> str:
    raw = os.environ.get(env_var, "").strip().rstrip("/")
    if not raw:
        return default_host
    # Reduce any base_url down to scheme://host[:port]; path is /v1/... in code.
    try:
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return raw


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return {"__error__": True, "status": exc.code, "detail": (detail or "")[:500]}
    except Exception as exc:
        return {"__error__": True, "status": 0, "detail": f"{exc}"[:500]}


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Hermes/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def _resolve_model(requested: str | None) -> str:
    if requested and requested in _MODELS:
        return requested
    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MiniMaxVideoGenProvider(VideoGenProvider):
    """MiniMax (Hailuo) video generation backend."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax (海螺)"

    def is_available(self) -> bool:
        return bool(_resolve_endpoint().get("api_key"))

    def list_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for mid, meta in _MODELS.items():
            # S2V-01 is subject-reference only (no plain T2V). Mark it as
            # image-only so the model picker doesn't suggest it for a prompt
            # without an image_url.
            if mid == "S2V-01":
                modalities: list[str] = ["image"]
            else:
                modalities = ["text"]
                if meta.get("supports_i2v") or meta.get("supports_subject_ref"):
                    modalities.append("image")
            out.append({
                "id": mid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta.get("price", "paid"),
                "modalities": modalities,
            })
        return out

    def default_model(self) -> str | None:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "MiniMax (海螺)",
            "badge": "paid",
            "tag": "Hailuo-2.3 / Hailuo-02 / S2V-01 — T2V + I2V async, CN endpoint by default",
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
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4"],
            "resolutions": ["768P", "1080P"],
            "max_duration": 10,
            "min_duration": 6,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "max_reference_images": 1,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        image_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        duration: int | None = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: str | None = None,
        audio: bool | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()

        ep = _resolve_endpoint()
        api_key = ep["api_key"]
        host = ep["host"]
        region = ep["region"]

        if not api_key:
            return error_response(
                error=(
                    "No MiniMax API key configured. Set MINIMAX_CN_API_KEY "
                    "(api.minimaxi.com) or MINIMAX_API_KEY (api.minimax.io)."
                ),
                error_type="auth_required",
                provider="minimax",
                prompt=prompt,
            )

        if not prompt:
            return error_response(
                error="Prompt is required.",
                error_type="missing_prompt",
                provider="minimax",
            )

        model_id = _resolve_model(model)
        meta = _MODELS[model_id]

        # Route by modality. image_url drives I2V; subject_reference uses
        # reference_image_urls and only S2V-01 supports it.
        image_url_norm = (image_url or "").strip() or None
        ref_list = [r.strip() for r in (reference_image_urls or []) if r and r.strip()]
        ref_first = ref_list[0] if ref_list else None

        aspect_norm = aspect_ratio if aspect_ratio in _VALID_ASPECTS else "16:9"
        # MiniMax uses uppercase 768P/1080P (not 720P — that was wrong).
        res_norm = (resolution or "768P").upper()
        if res_norm not in ("480P", "768P", "1080P"):
            res_norm = "768P"
        if model_id in ("S2V-01",):
            res_norm = "1080P"  # S2V-01 only supports 1080P per docs
        dur_norm = int(duration) if duration else 6
        if dur_norm not in meta["durations"]:
            dur_norm = min(meta["durations"], key=lambda d: abs(d - dur_norm))

        payload: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt[:2000],
            "duration": dur_norm,
            "resolution": res_norm,
        }

        modality_used = "text"
        if model_id == "S2V-01":
            # S2V-01: subject_reference, 6s only, 1080P only.
            if not ref_first:
                return error_response(
                    error=(
                        "S2V-01 requires a subject reference image. "
                        "Pass image_url=... (face photo) to use it."
                    ),
                    error_type="modality_unsupported",
                    provider="minimax",
                    model=model_id,
                    prompt=prompt,
                )
            payload["subject_reference"] = [
                {"type": "character", "image": [ref_first]}
            ]
            modality_used = "image"
        elif image_url_norm:
            if not meta.get("supports_i2v"):
                return error_response(
                    error=(
                        f"Model {model_id} does not support image-to-video. "
                        "Pass a text prompt only, or pick S2V-01 with image_url."
                    ),
                    error_type="modality_unsupported",
                    provider="minimax",
                    model=model_id,
                    prompt=prompt,
                )
            payload["first_frame_image"] = image_url_norm
            modality_used = "image"

        # Optional last_frame_image (Hailuo-02 supports it; we accept for any
        # I2V-capable model since docs only show it on Hailuo-02/2.3 explicitly).
        last_frame = kwargs.get("last_frame_image")
        if isinstance(last_frame, str) and last_frame.strip() and modality_used == "image":
            payload["last_frame_image"] = last_frame.strip()

        # 1. Submit
        submit = _request_json(
            "POST",
            f"{host}{VIDEO_GEN_PATH}",
            api_key,
            payload=payload,
            timeout=60,
        )
        if submit.get("__error__"):
            return error_response(
                error=f"MiniMax submit error {submit['status']}: {submit['detail']}",
                error_type="api_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
            )

        task_id = submit.get("task_id")
        # The docs also show a top-level {"task_id": "..."}; some regions
        # wrap it in {"data": {"task_id": ...}}. Handle both.
        if not task_id and isinstance(submit.get("data"), dict):
            task_id = submit["data"].get("task_id")
        if not task_id:
            base_resp = submit.get("base_resp") or {}
            msg = base_resp.get("status_msg") or json.dumps(submit)[:300]
            return error_response(
                error=f"MiniMax submit returned no task_id: {msg}",
                error_type="empty_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
            )

        logger.info("MiniMax video task submitted: %s (model=%s)", task_id, model_id)

        # 2. Poll
        file_id: str | None = None
        fail_msg: str | None = None
        deadline = time.monotonic() + POLL_TIMEOUT_SEC
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            query = _request_json(
                "GET",
                f"{host}{QUERY_PATH}?task_id={urllib.parse.quote(str(task_id))}",
                api_key,
                timeout=30,
            )
            if query.get("__error__"):
                # Transient — keep trying unless we're close to the deadline.
                logger.debug("MiniMax query transient error: %s", query)
                continue
            status = (query.get("status") or "").lower()
            if status == "success":
                file_id = query.get("file_id")
                if not file_id and isinstance(query.get("data"), dict):
                    file_id = query["data"].get("file_id")
                break
            if status in ("fail", "failed"):
                fail_msg = query.get("error_message") or query.get("error") or status
                if not fail_msg and isinstance(query.get("data"), dict):
                    fail_msg = query["data"].get("error_message")
                break
            # "processing" or empty -> keep polling.

        if not file_id:
            detail = fail_msg or f"task {task_id} did not complete within {POLL_TIMEOUT_SEC}s"
            return error_response(
                error=f"MiniMax video {detail}",
                error_type="api_error" if fail_msg else "timeout",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_norm,
                duration=dur_norm,
            )

        # 3. Resolve download URL
        retrieve = _request_json(
            "GET",
            f"{host}{FILES_PATH}?file_id={urllib.parse.quote(str(file_id))}",
            api_key,
            timeout=30,
        )
        if retrieve.get("__error__"):
            return error_response(
                error=f"MiniMax file retrieve error {retrieve['status']}: {retrieve['detail']}",
                error_type="api_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
            )

        download_url = None
        file_obj = retrieve.get("file")
        if isinstance(file_obj, dict):
            download_url = file_obj.get("download_url")
        if not download_url and isinstance(retrieve.get("data"), dict):
            file_obj = retrieve["data"].get("file")
            if isinstance(file_obj, dict):
                download_url = file_obj.get("download_url")
        if not download_url:
            return error_response(
                error="MiniMax file retrieve returned no download_url",
                error_type="empty_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
            )

        # 4. Download bytes -> cache
        try:
            raw = _download_bytes(download_url, timeout=180)
        except Exception as exc:
            return error_response(
                error=f"Could not download MiniMax video: {exc}",
                error_type="io_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
            )

        try:
            saved_path = save_bytes_video(raw, prefix=f"minimax_{model_id}", extension="mp4")
            video_ref = str(saved_path)
        except Exception as exc:
            # Last-ditch: hand back the remote URL so the user can still grab it.
            logger.warning("Could not cache MiniMax video locally: %s", exc)
            video_ref = download_url

        extra: dict[str, Any] = {
            "region": region,
            "task_id": task_id,
            "file_id": file_id,
            "resolution": res_norm,
        }
        if isinstance(file_obj, dict) and file_obj.get("file_size"):
            extra["file_size"] = file_obj["file_size"]

        return success_response(
            video=video_ref,
            model=model_id,
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=aspect_norm,
            duration=dur_norm,
            provider="minimax",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the MiniMax video generation provider."""
    ctx.register_video_gen_provider(MiniMaxVideoGenProvider())
