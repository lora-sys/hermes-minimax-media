"""Shared pytest fixtures for hermes-minimax-media.

The real `agent.image_gen_provider` and `agent.video_gen_provider` modules
import from inside the Hermes runtime, which is unavailable when this
package's tests run in a vanilla CI environment. We provide lightweight
stubs so the plugin modules can be imported for unit testing.

If the real modules ARE available (e.g. when running inside Hermes),
they take precedence.
"""
import sys
import types
from typing import Any

# --- helpers used by both provider modules ----------------------------------

DEFAULT_ASPECT_RATIO = "landscape"
DEFAULT_RESOLUTION = "768P"
VALID_ASPECT_RATIOS = ("landscape", "square", "portrait")


def _ensure_stub_module(name: str, attrs: dict[str, Any]) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# --- image_gen_provider stub -----------------------------------------------

def _make_image_stub():
    import abc

    class ImageGenProvider(abc.ABC):
        @property
        @abc.abstractmethod
        def name(self) -> str: ...

        @property
        def display_name(self) -> str:
            return self.name.title()

        def is_available(self) -> bool:
            return True

        def list_models(self) -> list[dict[str, Any]]:
            return []

        def default_model(self) -> str | None:
            return None

        def capabilities(self) -> dict[str, Any]:
            return {"modalities": ["text"], "max_reference_images": 0}

        def get_setup_schema(self) -> dict[str, Any]:
            return {"name": self.display_name, "badge": "", "tag": "", "env_vars": []}

        @abc.abstractmethod
        def generate(self, prompt, aspect_ratio=DEFAULT_ASPECT_RATIO, *, image_url=None,
                     reference_image_urls=None, **kwargs): ...

    def resolve_aspect_ratio(value):
        if value in VALID_ASPECT_RATIOS:
            return value
        return DEFAULT_ASPECT_RATIO

    def normalize_reference_images(value):
        if not value:
            return None
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if v]
        return [str(value)]

    def save_b64_image(b64, *, prefix="image", extension="png"):
        import base64
        from pathlib import Path
        out = Path(f"/tmp/{prefix}_stub.jpg")
        out.write_bytes(base64.b64decode(b64) if b64 else b"")
        return out

    def save_url_image(url, *, prefix="image"):
        from pathlib import Path
        out = Path(f"/tmp/{prefix}_stub_url.txt")
        out.write_text(url)
        return out

    def success_response(*, image, model, prompt, aspect_ratio, provider,
                        modality="text", extra=None):
        d = {"success": True, "image": image, "model": model, "prompt": prompt,
             "aspect_ratio": aspect_ratio, "provider": provider, "modality": modality}
        if extra:
            d.update(extra)
        return d

    def error_response(*, error, error_type="provider_error", provider="", model="",
                      prompt="", aspect_ratio=""):
        return {"success": False, "image": None, "error": error, "error_type": error_type,
                "model": model, "prompt": prompt, "aspect_ratio": aspect_ratio,
                "provider": provider}

    return dict(
        DEFAULT_ASPECT_RATIO=DEFAULT_ASPECT_RATIO,
        VALID_ASPECT_RATIOS=VALID_ASPECT_RATIOS,
        ImageGenProvider=ImageGenProvider,
        resolve_aspect_ratio=resolve_aspect_ratio,
        normalize_reference_images=normalize_reference_images,
        save_b64_image=save_b64_image,
        save_url_image=save_url_image,
        success_response=success_response,
        error_response=error_response,
    )


# --- video_gen_provider stub -----------------------------------------------

def _make_video_stub():
    import abc

    class VideoGenProvider(abc.ABC):
        @property
        @abc.abstractmethod
        def name(self) -> str: ...

        @property
        def display_name(self) -> str:
            return self.name.title()

        def is_available(self) -> bool:
            return True

        def list_models(self) -> list[dict[str, Any]]:
            return []

        def default_model(self) -> str | None:
            return None

        def capabilities(self) -> dict[str, Any]:
            return {"modalities": ["text"], "aspect_ratios": ["16:9"],
                    "resolutions": ["768P", "1080P"], "max_duration": 10,
                    "min_duration": 1, "supports_audio": False,
                    "supports_negative_prompt": False, "max_reference_images": 0}

        def get_setup_schema(self) -> dict[str, Any]:
            return {"name": self.display_name, "badge": "", "tag": "", "env_vars": []}

        @abc.abstractmethod
        def generate(self, prompt, *, model=None, image_url=None,
                     reference_image_urls=None, duration=None,
                     aspect_ratio=DEFAULT_ASPECT_RATIO,
                     resolution=DEFAULT_RESOLUTION,
                     negative_prompt=None, audio=None, seed=None, **kwargs): ...

    def save_bytes_video(raw, *, prefix="video", extension="mp4"):
        from pathlib import Path
        out = Path(f"/tmp/{prefix}_stub.{extension}")
        out.write_bytes(raw or b"")
        return out

    def success_response(*, video, model, prompt, modality="text", aspect_ratio="",
                        duration=0, provider, extra=None):
        d = {"success": True, "video": video, "model": model, "prompt": prompt,
             "modality": modality, "aspect_ratio": aspect_ratio, "duration": duration,
             "provider": provider}
        if extra:
            d.update(extra)
        return d

    def error_response(*, error, error_type="provider_error", provider="", model="",
                      prompt="", aspect_ratio="", duration=0):
        return {"success": False, "video": None, "error": error, "error_type": error_type,
                "model": model, "prompt": prompt, "aspect_ratio": aspect_ratio,
                "duration": duration, "provider": provider}

    return dict(
        DEFAULT_ASPECT_RATIO="16:9",
        DEFAULT_RESOLUTION=DEFAULT_RESOLUTION,
        VideoGenProvider=VideoGenProvider,
        save_bytes_video=save_bytes_video,
        success_response=success_response,
        error_response=error_response,
    )


# Install stubs only if the real modules aren't there.
try:
    import agent.image_gen_provider  # noqa: F401
except Exception:
    pkg = types.ModuleType("agent")
    pkg.__path__ = []  # mark as a package so submodule imports work
    sys.modules["agent"] = pkg
    _ensure_stub_module("agent.image_gen_provider", _make_image_stub())

try:
    import agent.video_gen_provider  # noqa: F401
except Exception:
    if "agent" not in sys.modules:
        pkg = types.ModuleType("agent")
        pkg.__path__ = []
        sys.modules["agent"] = pkg
    _ensure_stub_module("agent.video_gen_provider", _make_video_stub())
