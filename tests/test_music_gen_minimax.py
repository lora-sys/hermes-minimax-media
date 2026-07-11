"""Unit tests for the MiniMax (海螺) music generation plugin.

These tests do NOT hit the live API — they verify routing, env resolution,
payload construction, error responses, and model catalog correctness. To
smoke-test against a real MiniMax key, set MINIMAX_CN_API_KEY and run
`pytest tests/test_music_gen_minimax.py --run-live`.
"""
import base64
import importlib
import os

import pytest

music_mod = importlib.import_module("hermes_minimax_media.plugins.music_gen.minimax")
MusicProvider = music_mod  # module-level functions, not a class


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


class TestMusicEndpointResolution:
    def setup_method(self):
        self._saved = {k: os.environ.pop(k, None) for k in
                       ("MINIMAX_CN_API_KEY", "MINIMAX_API_KEY",
                        "MINIMAX_CN_BASE_URL", "MINIMAX_BASE_URL")}
        # Clean any leftover state.
        for k in list(self._saved):
            self._saved[k] = os.environ.pop(k, None)

    def teardown_method(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            elif os.environ.get(k) is not None:
                del os.environ[k]

    def test_no_keys_returns_empty_key(self):
        ep = music_mod._resolve_endpoint()
        assert ep["api_key"] == ""
        assert ep["region"] == "global"
        assert ep["host"] == music_mod.GLOBAL_HOST

    def test_cn_key_takes_priority(self):
        os.environ["MINIMAX_CN_API_KEY"] = "sk-cn"
        os.environ["MINIMAX_API_KEY"] = "sk-gl"
        ep = music_mod._resolve_endpoint()
        assert ep["api_key"] == "sk-cn"
        assert ep["region"] == "cn"
        assert ep["host"] == music_mod.CN_HOST

    def test_global_fallback(self):
        os.environ["MINIMAX_API_KEY"] = "sk-gl"
        ep = music_mod._resolve_endpoint()
        assert ep["api_key"] == "sk-gl"
        assert ep["region"] == "global"
        assert ep["host"] == music_mod.GLOBAL_HOST

    def test_cn_base_url_override(self):
        os.environ["MINIMAX_CN_API_KEY"] = "sk-cn"
        os.environ["MINIMAX_CN_BASE_URL"] = "https://api.minimaxi.com"
        ep = music_mod._resolve_endpoint()
        assert ep["host"] == "https://api.minimaxi.com"

    def test_cn_base_url_with_path_gets_reduced_to_host(self):
        os.environ["MINIMAX_CN_API_KEY"] = "sk-cn"
        os.environ["MINIMAX_CN_BASE_URL"] = "https://api.minimaxi.com/v1/music_generation"
        ep = music_mod._resolve_endpoint()
        # Path should be stripped — we re-add /v1/music_generation ourselves.
        assert ep["host"] == "https://api.minimaxi.com"


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


class TestPayloadConstruction:
    def test_default_vocals_mode(self):
        p = music_mod._build_payload(
            prompt="Indie folk, melancholic",
            lyrics="[Verse]\nA line",
            model="music-2.6",
            is_instrumental=False,
            lyrics_optimizer=False,
            output_format="url",
            sample_rate=44100,
            bitrate=256000,
            audio_format="mp3",
        )
        assert p["model"] == "music-2.6"
        assert p["prompt"] == "Indie folk, melancholic"
        assert p["lyrics"] == "[Verse]\nA line"
        assert "lyrics_optimizer" not in p
        assert "is_instrumental" not in p
        assert p["audio_setting"] == {
            "sample_rate": 44100,
            "bitrate": 256000,
            "format": "mp3",
        }
        assert p["output_format"] == "url"

    def test_instrumental_mode(self):
        p = music_mod._build_payload(
            prompt="Cinematic orchestral",
            lyrics=None,
            model="music-2.6",
            is_instrumental=True,
            lyrics_optimizer=False,
            output_format="url",
            sample_rate=44100,
            bitrate=256000,
            audio_format="mp3",
        )
        assert p["is_instrumental"] is True
        assert "lyrics" not in p

    def test_lyrics_optimizer_mode(self):
        p = music_mod._build_payload(
            prompt="A blues song about rain",
            lyrics=None,
            model="music-2.6",
            is_instrumental=False,
            lyrics_optimizer=True,
            output_format="url",
            sample_rate=44100,
            bitrate=256000,
            audio_format="mp3",
        )
        assert p["lyrics_optimizer"] is True
        assert p["lyrics"] == ""

    def test_prompt_is_truncated(self):
        long = "x" * 5000
        p = music_mod._build_payload(
            prompt=long,
            lyrics=None,
            model="music-2.6",
            is_instrumental=False,
            lyrics_optimizer=True,
            output_format="url",
            sample_rate=44100,
            bitrate=256000,
            audio_format="mp3",
        )
        assert len(p["prompt"]) == 2000


# ---------------------------------------------------------------------------
# generate_music() end-to-end (network mocked)
# ---------------------------------------------------------------------------


class TestGenerateMusic:
    def setup_method(self):
        self._saved = {k: os.environ.pop(k, None) for k in
                       ("MINIMAX_CN_API_KEY", "MINIMAX_API_KEY",
                        "MINIMAX_CN_BASE_URL", "MINIMAX_BASE_URL",
                        "HERMES_AUDIO_CACHE_DIR")}
        for k in list(self._saved):
            self._saved[k] = os.environ.pop(k, None)

    def teardown_method(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            elif os.environ.get(k) is not None:
                del os.environ[k]

    def test_empty_prompt_returns_invalid_argument(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        r = music_mod.generate_music(prompt="")
        assert r["success"] is False
        assert r["error_type"] == "invalid_argument"

    def test_no_key_returns_auth_required(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        r = music_mod.generate_music(prompt="a blues song")
        assert r["success"] is False
        assert r["error_type"] == "auth_required"
        assert r["provider"] == "minimax"

    def test_unknown_model_returns_invalid_argument(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        r = music_mod.generate_music(prompt="anything", model="gpt-music")
        assert r["success"] is False
        assert r["error_type"] == "invalid_argument"
        assert "music-2.6" in r["error"]

    def test_url_mode_writes_file_to_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        monkeypatch.setenv("HERMES_AUDIO_CACHE_DIR", str(tmp_path))

        # Fake the MiniMax POST -> URL response, and the download.
        monkeypatch.setattr(
            music_mod, "_post_json",
            lambda url, key, payload, timeout=300: {
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": "https://example.com/song.mp3"},
                "id": "req-abc",
            },
        )
        fake_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x21TPE1" + b"\x00" * 64
        monkeypatch.setattr(
            music_mod, "_download_bytes",
            lambda url, timeout=120: fake_bytes,
        )

        r = music_mod.generate_music(
            prompt="Soulful blues, rainy night",
            lyrics="[Verse]\nRain on the roof",
        )
        assert r["success"], r
        assert r["audio"] and os.path.isfile(r["audio"])
        assert r["audio"].endswith(".mp3")
        assert r["provider"] == "minimax"
        assert r["model"] == "music-2.6"
        assert r["extra"]["region"] == "cn"
        assert r["extra"]["request_id"] == "req-abc"
        assert r["lyrics"] == "[Verse]\nRain on the roof"

    def test_hex_mode_decodes_base64(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        monkeypatch.setenv("HERMES_AUDIO_CACHE_DIR", str(tmp_path))

        audio_bytes = b"\xff\xfb\x90\x00" + b"\x00" * 128  # fake MP3 header
        monkeypatch.setattr(
            music_mod, "_post_json",
            lambda url, key, payload, timeout=300: {
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": base64.b64encode(audio_bytes).decode("ascii")},
                "id": "req-hex",
            },
        )

        r = music_mod.generate_music(
            prompt="Ambient pad",
            output_format="hex",
            audio_format="mp3",
        )
        assert r["success"], r
        assert r["audio"] and os.path.isfile(r["audio"])
        with open(r["audio"], "rb") as f:
            assert f.read() == audio_bytes

    def test_api_error_response_propagates_status_code(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        monkeypatch.setattr(
            music_mod, "_post_json",
            lambda url, key, payload, timeout=300: {
                "base_resp": {"status_code": 1004, "status_msg": "invalid lyrics"},
                "data": {},
            },
        )
        r = music_mod.generate_music(prompt="a song", lyrics="bad")
        assert r["success"] is False
        assert r["error_type"] == "api_error"
        assert "1004" in r["error"]
        assert "invalid lyrics" in r["error"]

    def test_http_401_maps_to_auth_error(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-bad")
        monkeypatch.setattr(
            music_mod, "_post_json",
            lambda url, key, payload, timeout=300: {
                "__error__": True, "status": 401, "detail": "Unauthorized",
            },
        )
        r = music_mod.generate_music(prompt="a song")
        assert r["success"] is False
        assert r["error_type"] == "auth_error"
        assert "401" in r["error"]

    def test_http_429_maps_to_rate_limited(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        monkeypatch.setattr(
            music_mod, "_post_json",
            lambda url, key, payload, timeout=300: {
                "__error__": True, "status": 429, "detail": "balance not enough",
            },
        )
        r = music_mod.generate_music(prompt="a song")
        assert r["success"] is False
        assert r["error_type"] == "rate_limited"

    def test_empty_data_audio_returns_empty_response(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        monkeypatch.setattr(
            music_mod, "_post_json",
            lambda url, key, payload, timeout=300: {
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {},  # no audio
            },
        )
        r = music_mod.generate_music(prompt="a song")
        assert r["success"] is False
        assert r["error_type"] == "empty_response"

    def test_global_endpoint_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-gl")
        monkeypatch.setenv("HERMES_AUDIO_CACHE_DIR", str(tmp_path))

        captured = {}
        def fake_post(url, key, payload, timeout=300):
            captured["url"] = url
            captured["payload"] = payload
            return {
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": "https://x/y.mp3"},
            }
        monkeypatch.setattr(music_mod, "_post_json", fake_post)
        monkeypatch.setattr(music_mod, "_download_bytes", lambda url, timeout=120: b"\x00" * 16)

        r = music_mod.generate_music(prompt="global endpoint test")
        assert r["success"], r
        assert captured["url"].startswith("https://api.minimax.io/")
        assert r["extra"]["region"] == "global"

    def test_instrumental_and_optimizer_are_mutually_exclusive(self, monkeypatch, tmp_path):
        """is_instrumental wins; lyrics_optimizer is dropped."""
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        monkeypatch.setenv("HERMES_AUDIO_CACHE_DIR", str(tmp_path))
        captured = {}
        def fake_post(url, key, payload, timeout=300):
            captured["payload"] = payload
            return {"base_resp": {"status_code": 0, "status_msg": "success"},
                    "data": {"audio": "https://x/y.mp3"}}
        monkeypatch.setattr(music_mod, "_post_json", fake_post)
        monkeypatch.setattr(music_mod, "_download_bytes", lambda url, timeout=120: b"\x00" * 8)

        r = music_mod.generate_music(
            prompt="test", is_instrumental=True, lyrics_optimizer=True
        )
        assert r["success"]
        assert captured["payload"].get("is_instrumental") is True
        assert "lyrics_optimizer" not in captured["payload"]


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_schema_has_required_fields(self):
        schema = music_mod.MUSIC_GENERATE_TOOL_SCHEMA
        assert schema["name"] == "music_generate"
        assert "parameters" in schema
        props = schema["parameters"]["properties"]
        assert "prompt" in props
        assert "lyrics" in props
        assert "model" in props
        assert "is_instrumental" in props
        assert "lyrics_optimizer" in props
        # prompt is the only required field — lyrics is optional (instrumental
        # mode doesn't need them, lyrics_optimizer mode generates them).
        assert schema["parameters"]["required"] == ["prompt"]

    def test_schema_model_enum_includes_all_supported(self):
        schema = music_mod.MUSIC_GENERATE_TOOL_SCHEMA
        enum = schema["parameters"]["properties"]["model"]["enum"]
        assert "music-2.6" in enum
        assert "music-cover" in enum
        assert "music-cover-free" in enum

    def test_handler_dispatches_to_generate_music(self, monkeypatch):
        """Verify the handler unpacks the LLM ``args`` dict and forwards.

        Bug regression: the dispatcher calls ``handler(args, **kw)`` with the
        raw JSON object the LLM emitted — so the handler must read keys out
        of that dict. The original signature ``handler(prompt: str, *,
        lyrics=...)`` bound the entire dict to ``prompt`` and crashed inside
        ``prompt.strip()``. This test guards the fix.
        """
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        called = {}
        def fake_generate(**kwargs):
            called.update(kwargs)
            return {"success": True, "audio": "/tmp/x.mp3"}
        monkeypatch.setattr(music_mod, "generate_music", fake_generate)

        # The shape hermes-agent actually passes in: an args dict.
        r = music_mod._music_generate_handler(
            {"prompt": "hello", "lyrics": "[Verse]\nhi", "is_instrumental": True}
        )
        assert r["success"]
        assert called["prompt"] == "hello"
        assert called["lyrics"] == "[Verse]\nhi"
        assert called["is_instrumental"] is True

    def test_handler_applies_defaults_for_missing_args(self, monkeypatch):
        """When the LLM only sends ``prompt``, the handler fills in defaults
        for everything else — same behavior the JSON schema advertises."""
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        called = {}
        monkeypatch.setattr(
            music_mod, "generate_music",
            lambda **kw: called.update(kw) or {"success": True},
        )
        music_mod._music_generate_handler({"prompt": "only prompt"})
        assert called["prompt"] == "only prompt"
        assert called["lyrics"] is None
        assert called["is_instrumental"] is False
        assert called["sample_rate"] == 44100
        assert called["bitrate"] == 256000
        assert called["audio_format"] == "mp3"
        assert called["output_format"] == "url"

    def test_handler_coerces_string_bool_and_int_args(self, monkeypatch):
        """Some model providers serialize ``True`` as ``\"true\"`` and integers
        as strings in the tool-args payload. The handler must still produce
        the right typed values downstream."""
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        called = {}
        monkeypatch.setattr(
            music_mod, "generate_music",
            lambda **kw: called.update(kw) or {"success": True},
        )
        music_mod._music_generate_handler({
            "prompt": "p",
            "is_instrumental": "true",
            "lyrics_optimizer": "1",
            "sample_rate": "32000",
            "bitrate": "320000",
        })
        assert called["is_instrumental"] is True
        assert called["lyrics_optimizer"] is True
        assert called["sample_rate"] == 32000
        assert called["bitrate"] == 320000

    def test_handler_survives_none_args(self, monkeypatch):
        """Defensive: ``args=None`` shouldn't blow up the wrapper."""
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-test")
        called = {}
        def fake_gen(**kw):
            called.update(kw)
            # Mirror the real generate_music behavior for empty prompt.
            if not (kw.get("prompt") or "").strip():
                return {
                    "success": False,
                    "error": "prompt empty",
                    "error_type": "invalid_argument",
                }
            return {"success": True}
        monkeypatch.setattr(music_mod, "generate_music", fake_gen)
        r = music_mod._music_generate_handler(None)  # type: ignore[arg-type]
        # Should call generate_music with empty prompt (which then errors out
        # cleanly with invalid_argument), not crash on None.strip().
        assert called["prompt"] == ""
        assert r["success"] is False
        assert r["error_type"] == "invalid_argument"


# ---------------------------------------------------------------------------
# Optional live smoke (opt-in via --run-live)
# ---------------------------------------------------------------------------


@pytest.mark.requires_live
class TestLiveSmoke:
    def test_instrumental_short(self):
        if not os.environ.get("MINIMAX_CN_API_KEY") and not os.environ.get("MINIMAX_API_KEY"):
            pytest.skip("no MiniMax key configured")
        # Use instrumental + lyrics_optimizer to avoid hand-writing lyrics.
        r = music_mod.generate_music(
            prompt="Cheerful ukulele, 100 BPM, sunny morning",
            is_instrumental=True,
            audio_format="mp3",
        )
        assert r["success"], r
        assert r["audio"]
        # Cached file should exist locally.
        assert os.path.isfile(r["audio"])