"""MiniMax music generation backend.

Exposes MiniMax's music models as a Hermes ``music_generate`` tool.

- music-2.6:           Lyrics + style prompt -> full song with vocals (or
                       instrumental via ``is_instrumental=True``). Default.
- music-cover:         Two-step cover (requires ``cover_feature_id``).
- music-cover-free:    Same as music-cover, free tier.

API reference: https://platform.minimax.io/docs/api-reference/music-generation

Workflow (synchronous — unlike video, no task polling):
    1. POST /v1/music_generation  -> {data.audio (hex or url), base_resp}
    2. If hex: decode to bytes -> cache to ~/.hermes/audio_cache/
       If url: GET <url>          -> bytes -> cache

Env / base URL resolution (CN-first, global fallback):
    1. ``MINIMAX_CN_API_KEY`` + host ``https://api.minimaxi.com``  (CN)
    2. ``MINIMAX_API_KEY``   + host ``https://api.minimax.io``    (global)
    3. ``MINIMAX_CN_BASE_URL`` / ``MINIMAX_BASE_URL`` may override the host.

The music tool is registered as a standalone ``music_generate`` tool
(rather than going through a ``MusicGenProvider`` ABC) because Hermes does
not yet ship a music-gen provider interface. This mirrors the early days
of image/video gen, where the first provider shipped before the ABC.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

CN_HOST = "https://api.minimaxi.com"
GLOBAL_HOST = "https://api.minimax.io"
MUSIC_GEN_PATH = "/v1/music_generation"

DEFAULT_MODEL = "music-2.6"

_MODELS: dict[str, dict[str, Any]] = {
    "music-2.6": {
        "display": "Music 2.6",
        "speed": "~30-90s for a typical song",
        "strengths": "Latest flagship T2M (text-to-music) — vocals + instruments",
        "price": "paid",
        "supports_lyrics": True,
        "supports_instrumental": True,
        "requires_cover_feature_id": False,
    },
    "music-cover": {
        "display": "Music Cover (paid)",
        "speed": "~30-90s",
        "strengths": "Cover song workflow — uses cover_feature_id from preprocess",
        "price": "paid",
        "supports_lyrics": True,
        "supports_instrumental": False,
        "requires_cover_feature_id": True,
    },
    "music-cover-free": {
        "display": "Music Cover (free)",
        "speed": "~30-90s",
        "strengths": "Free-tier cover song",
        "price": "free",
        "supports_lyrics": True,
        "supports_instrumental": False,
        "requires_cover_feature_id": True,
    },
}

_VALID_BITRATES = {128000, 256000, 320000}
_VALID_SAMPLE_RATES = {24000, 32000, 44100}
_VALID_FORMATS = {"mp3", "wav", "pcm"}


# ---------------------------------------------------------------------------
# Endpoint / key resolution (shared contract with image/video plugins)
# ---------------------------------------------------------------------------


def _resolve_endpoint() -> dict[str, str]:
    """Return ``{"host": ..., "api_key": ..., "region": "cn"|"global"}``.

    CN takes priority when its key is set — matches the user's primary
    configuration. Falls back to global. Empty ``api_key`` if neither is
    set (caller turns this into an auth error).
    """
    cn_key = os.environ.get("MINIMAX_CN_API_KEY", "").strip()
    if cn_key:
        raw = os.environ.get("MINIMAX_CN_BASE_URL", "").strip().rstrip("/")
        if raw:
            try:
                parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(raw)
                if parsed.scheme and parsed.netloc:
                    host = f"{parsed.scheme}://{parsed.netloc}"
                else:
                    host = raw
            except Exception:
                host = raw
        else:
            host = CN_HOST
        return {"host": host, "api_key": cn_key, "region": "cn"}

    gl_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if gl_key:
        raw = os.environ.get("MINIMAX_BASE_URL", "").strip().rstrip("/")
        if raw:
            try:
                parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(raw)
                if parsed.scheme and parsed.netloc:
                    host = f"{parsed.scheme}://{parsed.netloc}"
                else:
                    host = raw
            except Exception:
                host = raw
        else:
            host = GLOBAL_HOST
        return {"host": host, "api_key": gl_key, "region": "global"}

    return {"host": GLOBAL_HOST, "api_key": "", "region": "global"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 300,
) -> dict[str, Any]:
    """POST JSON, return parsed response or ``{"__error__": ...}``."""
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
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        # Try to extract the MiniMax base_resp.status_msg for cleaner errors.
        try:
            detail_json = json.loads(detail)
            detail = detail_json.get("base_resp", {}).get("status_msg") or detail
        except Exception:
            pass
        return {"__error__": True, "status": exc.code, "detail": (detail or "")[:500]}
    except Exception as exc:
        return {"__error__": True, "status": 0, "detail": f"{exc}"[:500]}


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Hermes/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Audio cache
# ---------------------------------------------------------------------------


def _cache_dir() -> str:
    """Return and ensure the local audio cache directory."""
    base = os.environ.get("HERMES_AUDIO_CACHE_DIR")
    if not base:
        home = os.path.expanduser("~")
        base = os.path.join(home, ".hermes", "audio_cache")
    os.makedirs(base, exist_ok=True)
    return base


def _save_audio_bytes(data: bytes, *, prefix: str, ext: str = "mp3") -> str:
    """Save bytes to the cache directory and return the absolute path."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.{ext}"
    path = os.path.join(_cache_dir(), fname)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------------
# Core generation logic (testable in isolation)
# ---------------------------------------------------------------------------


def _build_payload(
    *,
    prompt: str,
    lyrics: str | None,
    model: str,
    is_instrumental: bool,
    lyrics_optimizer: bool,
    output_format: str,
    sample_rate: int,
    bitrate: int,
    audio_format: str,
) -> dict[str, Any]:
    """Construct the MiniMax /v1/music_generation payload."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt[:2000],
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "format": audio_format,
        },
        "output_format": output_format,
    }
    if is_instrumental:
        # MiniMax docs: is_instrumental=true implies no vocals.
        payload["is_instrumental"] = True
    elif lyrics_optimizer:
        # When true, model writes lyrics from prompt if lyrics is empty.
        payload["lyrics_optimizer"] = True
        payload["lyrics"] = lyrics or ""
    else:
        payload["lyrics"] = lyrics or ""
    return payload


def generate_music(
    *,
    prompt: str,
    lyrics: str | None = None,
    model: str | None = None,
    is_instrumental: bool = False,
    lyrics_optimizer: bool = False,
    output_format: str = "url",
    sample_rate: int = 44100,
    bitrate: int = 256000,
    audio_format: str = "mp3",
) -> dict[str, Any]:
    """Call the MiniMax Music API and return a result dict.

    Returns a dict mirroring the image/video providers' ``success_response``
    / ``error_response`` shape:

        success         bool
        audio           str | None   absolute path to saved file
        model           str
        prompt          str
        lyrics          str          (echoed)
        provider        str          "minimax"
        error           str          (only when success=False)
        error_type      str          (only when success=False)
        extra           dict         region, request_id, audio_format, ...
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return {
            "success": False,
            "error": "Prompt is required and must be a non-empty string",
            "error_type": "invalid_argument",
            "provider": "minimax",
        }

    # Resolve model.
    model_id = (model or "").strip() or DEFAULT_MODEL
    if model_id not in _MODELS:
        return {
            "success": False,
            "error": f"Unknown model '{model_id}'. Supported: {list(_MODELS)}",
            "error_type": "invalid_argument",
            "provider": "minimax",
            "model": model_id,
        }
    meta = _MODELS[model_id]

    # Normalize audio params.
    if sample_rate not in _VALID_SAMPLE_RATES:
        sample_rate = 44100
    if bitrate not in _VALID_BITRATES:
        bitrate = 256000
    if audio_format not in _VALID_FORMATS:
        audio_format = "mp3"
    if output_format not in ("url", "hex"):
        output_format = "url"

    # Instrumental and lyrics_optimizer are mutually exclusive (per docs).
    if is_instrumental and lyrics_optimizer:
        lyrics_optimizer = False

    ep = _resolve_endpoint()
    api_key = ep["api_key"]
    host = ep["host"]
    region = ep["region"]

    if not api_key:
        return {
            "success": False,
            "error": (
                "No MiniMax API key configured. Set MINIMAX_CN_API_KEY "
                "(api.minimaxi.com) or MINIMAX_API_KEY (api.minimax.io)."
            ),
            "error_type": "auth_required",
            "provider": "minimax",
            "model": model_id,
        }

    url = f"{host.rstrip('/')}{MUSIC_GEN_PATH}"
    payload = _build_payload(
        prompt=prompt,
        lyrics=lyrics,
        model=model_id,
        is_instrumental=is_instrumental,
        lyrics_optimizer=lyrics_optimizer,
        output_format=output_format,
        sample_rate=sample_rate,
        bitrate=bitrate,
        audio_format=audio_format,
    )

    resp = _post_json(url, api_key, payload, timeout=300)
    if resp.get("__error__"):
        status = resp.get("status", 0)
        detail = resp.get("detail", "")
        err_type = "api_error"
        if status in (401, 403):
            err_type = "auth_error"
        elif status == 429:
            err_type = "rate_limited"
        return {
            "success": False,
            "error": f"MiniMax music API error {status}: {detail}",
            "error_type": err_type,
            "provider": "minimax",
            "model": model_id,
            "prompt": prompt,
        }

    base_resp = resp.get("base_resp") or {}
    if base_resp.get("status_code", 0) != 0:
        return {
            "success": False,
            "error": (
                f"MiniMax error {base_resp.get('status_code')}: "
                f"{base_resp.get('status_msg', '(no message)')}"
            ),
            "error_type": "api_error",
            "provider": "minimax",
            "model": model_id,
            "prompt": prompt,
        }

    data = resp.get("data") or {}
    audio_field = data.get("audio")
    if not audio_field:
        return {
            "success": False,
            "error": f"MiniMax returned no audio data: {json.dumps(resp)[:300]}",
            "error_type": "empty_response",
            "provider": "minimax",
            "model": model_id,
            "prompt": prompt,
        }

    # Decode to bytes.
    try:
        if output_format == "hex":
            audio_bytes = base64.b64decode(audio_field)
        else:
            audio_bytes = _download_bytes(audio_field, timeout=180)
    except Exception as exc:
        logger.warning("Could not fetch MiniMax music bytes: %s", exc)
        # Hand back the remote URL as a last-ditch fallback.
        if output_format == "url":
            return {
                "success": True,
                "audio": audio_field,
                "model": model_id,
                "prompt": prompt,
                "lyrics": lyrics or "",
                "provider": "minimax",
                "extra": {"region": region, "audio_format": audio_format,
                          "remote_only": True},
            }
        return {
            "success": False,
            "error": f"Could not decode audio: {exc}",
            "error_type": "io_error",
            "provider": "minimax",
            "model": model_id,
            "prompt": prompt,
        }

    try:
        saved_path = _save_audio_bytes(
            audio_bytes, prefix=f"minimax_{model_id}", ext=audio_format
        )
        audio_ref = saved_path
    except Exception as exc:
        logger.warning("Could not cache MiniMax music locally: %s", exc)
        # If we asked for hex but couldn't cache, return raw base64 — caller
        # can save it elsewhere. For URL mode, the URL is still useful.
        if output_format == "hex":
            audio_ref = audio_field  # base64 string
        else:
            audio_ref = audio_field  # remote URL

    extra: dict[str, Any] = {
        "region": region,
        "audio_format": audio_format,
        "sample_rate": sample_rate,
        "bitrate": bitrate,
        "model_meta": meta,
    }
    if data.get("duration"):
        extra["duration_estimate"] = data["duration"]
    if resp.get("id"):
        extra["request_id"] = resp["id"]

    return {
        "success": True,
        "audio": audio_ref,
        "model": model_id,
        "prompt": prompt,
        "lyrics": lyrics or "",
        "provider": "minimax",
        "extra": extra,
    }


# ---------------------------------------------------------------------------
# Tool handler (bridges the dict result -> the tool's return contract)
# ---------------------------------------------------------------------------


def _music_generate_handler(
    prompt: str = "",
    *,
    lyrics: str | None = None,
    model: str | None = None,
    is_instrumental: bool = False,
    lyrics_optimizer: bool = False,
    output_format: str = "url",
    sample_rate: int = 44100,
    bitrate: int = 256000,
    audio_format: str = "mp3",
    **_: Any,
) -> dict[str, Any]:
    """Tool handler — delegates to :func:`generate_music`."""
    return generate_music(
        prompt=prompt,
        lyrics=lyrics,
        model=model,
        is_instrumental=bool(is_instrumental),
        lyrics_optimizer=bool(lyrics_optimizer),
        output_format=output_format,
        sample_rate=int(sample_rate),
        bitrate=int(bitrate),
        audio_format=audio_format,
    )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


MUSIC_GENERATE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "music_generate",
    "description": (
        "Generate a song from lyrics + a style prompt via the MiniMax Music API. "
        "Returns the absolute path to a saved MP3 (or WAV/PCM) file in "
        "~/.hermes/audio_cache/. Supports instrumental-only generation "
        "(is_instrumental=true) and lyrics-from-prompt (lyrics_optimizer=true, "
        "skip the lyrics arg). For cover songs, use model='music-cover' with "
        "cover_feature_id (not yet exposed here — use the API directly)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Style / mood / instrumentation description, e.g. "
                    "'Soulful Blues, Rainy Night, Electric Guitar, Male Vocal, Slow'."
                ),
            },
            "lyrics": {
                "type": "string",
                "description": (
                    "Song lyrics with structural tags like [Verse], [Chorus], "
                    "[Bridge], [Intro], [Outro]. Omit when lyrics_optimizer=true."
                ),
            },
            "model": {
                "type": "string",
                "enum": list(_MODELS),
                "default": DEFAULT_MODEL,
                "description": "Which MiniMax music model to use.",
            },
            "is_instrumental": {
                "type": "boolean",
                "default": False,
                "description": "true = generate instrumental (no vocals).",
            },
            "lyrics_optimizer": {
                "type": "boolean",
                "default": False,
                "description": "true = let the model write lyrics from the prompt.",
            },
            "output_format": {
                "type": "string",
                "enum": ["url", "hex"],
                "default": "url",
                "description": (
                    "Internal — 'url' lets MiniMax host a 24h URL; "
                    "'hex' returns base64 in the response (slower for large songs)."
                ),
            },
            "audio_format": {
                "type": "string",
                "enum": sorted(_VALID_FORMATS),
                "default": "mp3",
            },
            "sample_rate": {
                "type": "integer",
                "enum": sorted(_VALID_SAMPLE_RATES),
                "default": 44100,
            },
            "bitrate": {
                "type": "integer",
                "enum": sorted(_VALID_BITRATES),
                "default": 256000,
            },
        },
        "required": ["prompt"],
    },
}


def register(ctx) -> None:
    """Register the MiniMax music_generate tool.

    The tool is registered directly (not via a provider ABC) because Hermes
    does not yet ship a music_gen provider surface. If/when a music_gen
    provider ABC lands in ``hermes-agent``, this plugin can migrate to it
    without changing the public tool signature.
    """
    ctx.register_tool(
        name="music_generate",
        toolset="media",
        schema=MUSIC_GENERATE_TOOL_SCHEMA,
        handler=_music_generate_handler,
        requires_env=["MINIMAX_CN_API_KEY"],  # also accepts MINIMAX_API_KEY
        description=(
            "Generate music from lyrics + style via MiniMax Music API "
            "(music-2.6 / music-cover / music-cover-free)"
        ),
        emoji="🎵",
    )