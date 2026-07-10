"""Unit tests for the MiniMax (海螺) image + video plugins.

These tests do NOT hit the live API — they only verify routing, env
resolution, error responses, and model catalog correctness. To smoke-test
against a real MiniMax key, set MINIMAX_CN_API_KEY and run
`pytest tests/test_minimax.py --run-live` (we add a marker for that below).
"""
import importlib
import os

import pytest

# ---------------------------------------------------------------------------
# Imports — the conftest ensures the agent.* provider modules exist.
# ---------------------------------------------------------------------------

img_mod = importlib.import_module("hermes_minimax_media.plugins.image_gen.minimax")
vid_mod = importlib.import_module("hermes_minimax_media.plugins.video_gen.minimax")

ImageProvider = img_mod.MiniMaxImageGenProvider
VideoProvider = vid_mod.MiniMaxVideoGenProvider


# ---------------------------------------------------------------------------
# Image plugin
# ---------------------------------------------------------------------------

class TestImageEndpointResolution:
    def setup_method(self):
        self._saved = {k: os.environ.pop(k, None) for k in
                       ("MINIMAX_CN_API_KEY", "MINIMAX_API_KEY",
                        "MINIMAX_CN_BASE_URL", "MINIMAX_BASE_URL")}
        # Wipe any left-over values.
        for k in list(self._saved):
            self._saved[k] = os.environ.pop(k, None)

    def teardown_method(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            elif os.environ.get(k) is not None:
                del os.environ[k]

    def test_no_keys_returns_empty_key(self):
        ep = img_mod._resolve_endpoint()
        assert ep["api_key"] == ""
        assert ep["region"] == "global"
        assert ep["base_url"].endswith("/v1/image_generation")

    def test_cn_key_takes_priority(self):
        os.environ["MINIMAX_CN_API_KEY"] = "sk-cn"
        os.environ["MINIMAX_API_KEY"] = "sk-gl"
        ep = img_mod._resolve_endpoint()
        assert ep["api_key"] == "sk-cn"
        assert ep["region"] == "cn"
        assert "minimaxi.com" in ep["base_url"]

    def test_global_fallback(self):
        os.environ["MINIMAX_API_KEY"] = "sk-gl"
        ep = img_mod._resolve_endpoint()
        assert ep["api_key"] == "sk-gl"
        assert ep["region"] == "global"
        assert "api.minimax.io" in ep["base_url"]

    def test_cn_base_url_override(self):
        os.environ["MINIMAX_CN_API_KEY"] = "sk-cn"
        os.environ["MINIMAX_CN_BASE_URL"] = "https://api.minimaxi.com"
        ep = img_mod._resolve_endpoint()
        assert ep["base_url"] == "https://api.minimaxi.com/v1/image_generation"


class TestImageProvider:
    def test_is_available(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        p = ImageProvider()
        assert p.is_available() is True

    def test_models_catalog(self):
        p = ImageProvider()
        models = p.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "image-01"
        assert "modalities" not in models[0]  # image plugin doesn't use that

    def test_capabilities(self):
        p = ImageProvider()
        caps = p.capabilities()
        assert "text" in caps["modalities"]
        assert "image" in caps["modalities"]
        assert caps["max_reference_images"] == 1

    def test_no_key_returns_auth_required(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        p = ImageProvider()
        r = p.generate("a red apple", aspect_ratio="square")
        assert r["success"] is False
        assert r["error_type"] == "auth_required"
        assert r["provider"] == "minimax"

    def test_empty_prompt_returns_invalid_argument(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        p = ImageProvider()
        r = p.generate("", aspect_ratio="square")
        assert r["success"] is False
        assert r["error_type"] == "invalid_argument"

    def test_aspect_ratio_passthrough_to_payload(self, monkeypatch):
        """Verify the aspect_ratio 'square' -> '1:1' mapping."""
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        captured = {}

        def fake_post(url, key, payload, timeout=120):
            captured["url"] = url
            captured["key"] = key
            captured["payload"] = payload
            return {"__error__": True, "status": 500, "detail": "test"}

        monkeypatch.setattr(img_mod, "_post_json", fake_post)
        p = ImageProvider()
        p.generate("hello", aspect_ratio="square")
        assert captured["payload"]["aspect_ratio"] == "1:1"
        assert captured["payload"]["model"] == "image-01"
        assert captured["payload"]["response_format"] == "url"
        assert "subject_reference" not in captured["payload"]

    def test_image_url_adds_subject_reference(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        captured = {}

        def fake_post(url, key, payload, timeout=120):
            captured["payload"] = payload
            return {"__error__": True, "status": 500, "detail": "test"}

        monkeypatch.setattr(img_mod, "_post_json", fake_post)
        p = ImageProvider()
        p.generate("make it bigger", image_url="https://x/y.jpg", aspect_ratio="landscape")
        sr = captured["payload"]["subject_reference"]
        assert sr[0]["type"] == "character"
        assert sr[0]["image_file"] == "https://x/y.jpg"
        assert captured["payload"]["aspect_ratio"] == "16:9"


# ---------------------------------------------------------------------------
# Video plugin
# ---------------------------------------------------------------------------

class TestVideoEndpointResolution:
    def setup_method(self):
        self._saved = {k: os.environ.pop(k, None) for k in
                       ("MINIMAX_CN_API_KEY", "MINIMAX_API_KEY",
                        "MINIMAX_CN_BASE_URL", "MINIMAX_BASE_URL")}

    def teardown_method(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            elif os.environ.get(k) is not None:
                del os.environ[k]

    def test_cn_priority(self):
        os.environ["MINIMAX_CN_API_KEY"] = "sk-cn"
        os.environ["MINIMAX_API_KEY"] = "sk-gl"
        ep = vid_mod._resolve_endpoint()
        assert ep["region"] == "cn"
        assert ep["host"] == "https://api.minimaxi.com"

    def test_global_fallback(self):
        os.environ["MINIMAX_API_KEY"] = "sk-gl"
        ep = vid_mod._resolve_endpoint()
        assert ep["region"] == "global"
        assert ep["host"] == "https://api.minimax.io"


class TestVideoProvider:
    def test_models_catalog(self):
        p = VideoProvider()
        models = p.list_models()
        ids = [m["id"] for m in models]
        assert "MiniMax-Hailuo-2.3" in ids
        assert "MiniMax-Hailuo-02" in ids
        assert "S2V-01" in ids
        s2v = next(m for m in models if m["id"] == "S2V-01")
        # S2V-01 is subject-reference only.
        assert "image" in s2v["modalities"]
        assert "text" not in s2v["modalities"]
        hailuo = next(m for m in models if m["id"] == "MiniMax-Hailuo-2.3")
        assert "text" in hailuo["modalities"]
        assert "image" in hailuo["modalities"]

    def test_capabilities(self):
        p = VideoProvider()
        caps = p.capabilities()
        assert "768P" in caps["resolutions"]
        assert "1080P" in caps["resolutions"]
        assert "720P" not in caps["resolutions"]  # we explicitly don't advertise this

    def test_no_key_returns_auth_required(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        p = VideoProvider()
        r = p.generate("a sunset", model="MiniMax-Hailuo-2.3")
        assert r["success"] is False
        assert r["error_type"] == "auth_required"

    def test_empty_prompt(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        p = VideoProvider()
        r = p.generate("", model="MiniMax-Hailuo-2.3")
        assert r["success"] is False
        assert r["error_type"] == "missing_prompt"

    def test_s2v_requires_image(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        p = VideoProvider()
        r = p.generate("a cat walking", model="S2V-01")
        assert r["success"] is False
        assert r["error_type"] == "modality_unsupported"

    def test_hailuo23_rejects_image_url(self, monkeypatch):
        """Hailuo-2.3 DOES support I2V, so this should NOT error."""
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        # Stub the network — we only want to verify routing, not full submit.
        monkeypatch.setattr(vid_mod, "_request_json",
                            lambda *a, **kw: {"task_id": "x"})
        monkeypatch.setattr(vid_mod.time, "sleep", lambda *a, **kw: None)
        # Make polling return success immediately.
        def fake_query(*a, **kw):
            return {"status": "Success", "file_id": "f1"}
        monkeypatch.setattr(vid_mod, "_request_json", fake_query)
        # Then file retrieve returns a download url.
        responses = iter([
            {"task_id": "x"},  # submit
            {"status": "Success", "file_id": "f1"},  # query
            {"file": {"download_url": "https://x/y.mp4"}},  # retrieve
        ])
        monkeypatch.setattr(vid_mod, "_request_json",
                            lambda *a, **kw: next(responses))
        monkeypatch.setattr(vid_mod, "_download_bytes",
                            lambda url, timeout=120: b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
        p = VideoProvider()
        r = p.generate("a cat walking", model="MiniMax-Hailuo-2.3",
                      image_url="https://example.com/seed.jpg")
        assert r["success"], r
        assert r["modality"] == "image"
        assert r["model"] == "MiniMax-Hailuo-2.3"

    def test_resolution_normalization_768p(self, monkeypatch):
        """720P input gets coerced to 768P per docs."""
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        captured = {}
        # Mock _request_json to capture the submit payload and return
        # canned poll/retrieve responses so the full flow completes.
        responses = iter([
            {"task_id": "x"},
            {"status": "Success", "file_id": "f1"},
            {"file": {"download_url": "https://x/y.mp4"}},
        ])

        def fake_request_json(method, url, api_key, payload=None, timeout=30):
            if method == "POST":
                captured["payload"] = payload
            return next(responses)

        monkeypatch.setattr(vid_mod, "_request_json", fake_request_json)
        monkeypatch.setattr(vid_mod, "_download_bytes",
                            lambda url, timeout=120: b"\x00" * 32)
        monkeypatch.setattr(vid_mod.time, "sleep", lambda *a, **kw: None)
        p = VideoProvider()
        p.generate("x", model="MiniMax-Hailuo-2.3", resolution="720P")
        assert captured["payload"]["resolution"] == "768P"


# ---------------------------------------------------------------------------
# Optional live smoke (opt-in via --run-live)
# ---------------------------------------------------------------------------

@pytest.mark.requires_live
class TestLiveSmoke:
    def test_image_t2i(self):
        if not os.environ.get("MINIMAX_CN_API_KEY") and not os.environ.get("MINIMAX_API_KEY"):
            pytest.skip("no MiniMax key configured")
        p = ImageProvider()
        r = p.generate("a single red circle on white background", aspect_ratio="square")
        assert r["success"], r
        assert r["image"] and r["provider"] == "minimax"
