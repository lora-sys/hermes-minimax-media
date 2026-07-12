"""Tests for hermes-minimax-media's setup helper + entry-point wiring.

These tests run without any Hermes runtime or live API. They verify:

1. The ``hermes-minimax-setup`` CLI correctly writes ``plugins.enabled``
   and provider blocks into a target ``config.yaml``.
2. The CLI is idempotent — re-running on an already-configured file is a no-op.
3. ``--check`` returns the right exit code (0 configured, 1 not configured).
4. ``--uninstall`` cleanly reverses the setup.
5. The package declares the three expected entry points in the
   ``hermes_agent.plugins`` group (defense in depth — if pyproject.toml ever
   drops one, the corresponding provider silently stops loading).
6. Each plugin module's ``register()`` actually wires the right provider
   into a ``PluginContext`` stub.

Why this exists: prior to v0.3.0, the README instructed users to add
``video_gen/minimax`` (path-derived) to ``plugins.enabled``, which did NOT
match the pip-installed entry-point key (``minimax-vidgen``). Users who
followed the README got a silent no-op. The setup helper + these tests make
that class of bug impossible to reintroduce.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# Make the package importable without installing.
sys.path.insert(0, str(SRC_DIR))

setup = importlib.import_module("hermes_minimax_media.setup")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point setup at a temporary HERMES_HOME so tests don't touch real config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_config(home: Path, content: str) -> None:
    (home / "config.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry-point declaration
# ---------------------------------------------------------------------------


REQUIRED_ENTRY_POINTS = {"minimax-imggen", "minimax-vidgen", "minimax-musicgen"}


class TestEntryPoints:
    """The package must declare exactly the expected entry points. If a
    future refactor renames or drops one, these tests fail loud."""

    def test_all_three_entry_points_present(self) -> None:
        eps = importlib.metadata.entry_points()
        group = (
            eps.select(group="hermes_agent.plugins")
            if hasattr(eps, "select")
            else eps.get("hermes_agent.plugins", [])
        )
        names = {ep.name for ep in group}
        missing = REQUIRED_ENTRY_POINTS - names
        assert not missing, (
            f"pyproject.toml is missing entry points: {missing}. "
            f"Found: {sorted(names)}"
        )

    @pytest.mark.parametrize(
        "entry_name,module_marker",
        [
            ("minimax-imggen", "image_gen"),
            ("minimax-vidgen", "video_gen"),
            ("minimax-musicgen", "music_gen"),
        ],
    )
    def test_entry_point_targets_correct_module(
        self, entry_name: str, module_marker: str
    ) -> None:
        eps = importlib.metadata.entry_points()
        group = (
            eps.select(group="hermes_agent.plugins")
            if hasattr(eps, "select")
            else eps.get("hermes_agent.plugins", [])
        )
        ep = next((e for e in group if e.name == entry_name), None)
        assert ep is not None
        assert module_marker in ep.value, (
            f"{entry_name} should target a {module_marker} module, "
            f"got {ep.value!r}"
        )


# ---------------------------------------------------------------------------
# setup.apply_setup / apply_uninstall
# ---------------------------------------------------------------------------


class TestApplySetup:
    def test_adds_all_entry_points_to_empty_config(self, tmp_hermes_home: Path) -> None:
        cfg: dict[str, Any] = {}
        diff = setup.apply_setup(cfg)
        assert set(diff["added_to_enabled"]) == REQUIRED_ENTRY_POINTS
        assert set(diff["wrote_provider_blocks"]) == {"image_gen", "video_gen", "music_gen"}

        plugins = cfg["plugins"]
        assert set(plugins["enabled"]) == REQUIRED_ENTRY_POINTS

        assert cfg["image_gen"]["provider"] == "minimax"
        assert cfg["image_gen"]["minimax"]["model"] == "image-01"
        assert cfg["video_gen"]["provider"] == "minimax"
        assert cfg["video_gen"]["minimax"]["model"] == "MiniMax-Hailuo-2.3"
        assert cfg["music_gen"]["minimax"]["model"] == "music-2.6"

    def test_idempotent(self, tmp_hermes_home: Path) -> None:
        cfg: dict[str, Any] = {}
        setup.apply_setup(cfg)
        first = {
            "plugins": sorted(cfg["plugins"]["enabled"]),
            "image": cfg["image_gen"],
            "video": cfg["video_gen"],
            "music": cfg["music_gen"],
        }
        diff2 = setup.apply_setup(cfg)
        # Second run is a no-op for already-set values.
        assert diff2["added_to_enabled"] == []
        assert diff2["wrote_provider_blocks"] == []
        assert first == {
            "plugins": sorted(cfg["plugins"]["enabled"]),
            "image": cfg["image_gen"],
            "video": cfg["video_gen"],
            "music": cfg["music_gen"],
        }

    def test_preserves_other_enabled_plugins(self, tmp_hermes_home: Path) -> None:
        cfg = {
            "plugins": {
                "enabled": ["other-plugin", "another"],
            },
        }
        setup.apply_setup(cfg)
        enabled = cfg["plugins"]["enabled"]
        assert "other-plugin" in enabled
        assert "another" in enabled
        assert REQUIRED_ENTRY_POINTS.issubset(set(enabled))

    def test_does_not_overwrite_existing_provider_choice(self, tmp_hermes_home: Path) -> None:
        """If user already set image_gen.provider = openai (or another model
        under the minimax block), don't clobber it."""
        cfg = {
            "image_gen": {
                "provider": "openai",
                "openai": {"model": "gpt-image-2"},
            },
        }
        setup.apply_setup(cfg)
        # Untouched.
        assert cfg["image_gen"]["provider"] == "openai"
        assert cfg["image_gen"]["openai"]["model"] == "gpt-image-2"
        # But the minimax block was added so it's pre-wired if user switches.
        assert cfg["image_gen"]["minimax"]["model"] == "image-01"


class TestApplyUninstall:
    def test_removes_all_three_entry_points(self, tmp_hermes_home: Path) -> None:
        cfg: dict[str, Any] = {}
        setup.apply_setup(cfg)
        diff = setup.apply_uninstall(cfg)
        assert set(diff["removed_from_enabled"]) == REQUIRED_ENTRY_POINTS
        assert cfg["plugins"]["enabled"] == []
        # Provider blocks are gone, top-level keys removed too.
        assert "image_gen" not in cfg
        assert "video_gen" not in cfg
        assert "music_gen" not in cfg

    def test_preserves_non_minimax_provider_blocks(self, tmp_hermes_home: Path) -> None:
        cfg = {
            "image_gen": {"provider": "openai", "openai": {"model": "gpt-image-2"}},
        }
        diff = setup.apply_uninstall(cfg)
        # Uninstall leaves non-MiniMax provider blocks alone.
        assert cfg["image_gen"]["provider"] == "openai"
        assert diff["cleared_provider_blocks"] == []


# ---------------------------------------------------------------------------
# current_status / --check
# ---------------------------------------------------------------------------


class TestStatus:
    def test_not_configured_when_empty(self, tmp_hermes_home: Path) -> None:
        status = setup.current_status({})
        assert status["fully_configured"] is False
        assert set(status["missing_entry_points"]) == REQUIRED_ENTRY_POINTS

    def test_configured_after_apply(self, tmp_hermes_home: Path) -> None:
        cfg: dict[str, Any] = {}
        setup.apply_setup(cfg)
        status = setup.current_status(cfg)
        assert status["fully_configured"] is True
        assert status["missing_entry_points"] == []
        assert status["image_gen_provider"] == "minimax"
        assert status["video_gen_provider"] == "minimax"
        assert status["music_gen_minimax_model"] == "music-2.6"


# ---------------------------------------------------------------------------
# CLI integration (subprocess)
# ---------------------------------------------------------------------------



class TestCLI:
    def _run(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        # Load setup.py as a top-level module rather than as a submodule of
        # the hermes_minimax_media package. Why: importing the package
        # triggers the plugin subpackages' top-level `__init__.py`
        # (those `from agent.image_gen_provider import ...` and crash in CI).
        # Loading setup.py directly avoids the package import and faithfully
        # exercises what the `hermes-minimax-setup` console_script entry
        # point runs on a user machine (where hermes-agent IS installed
        # but the entry point itself never imports the package top-level).
        cmd = (
            "import sys, runpy; "
            "sys.path.insert(0, " + repr(str(SRC_DIR)) + "); "
            "ns = runpy.run_path(" + repr(str(SRC_DIR / "setup.py")) + "); "
            "sys.exit(ns['main'](sys.argv[1:]))"
        )
        return subprocess.run(
            [sys.executable, "-c", cmd, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        )

    def test_check_exits_nonzero_when_unconfigured(self, tmp_hermes_home: Path) -> None:
        result = self._run("--check")
        assert result.returncode == 1
        assert "NOT CONFIGURED" in result.stdout

    def test_check_exits_zero_after_setup(self, tmp_hermes_home: Path) -> None:
        self._run()  # default = apply
        result = self._run("--check")
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_dry_run_does_not_modify_file(self, tmp_hermes_home: Path) -> None:
        config = tmp_hermes_home / "config.yaml"
        assert not config.exists()
        result = self._run("--print")
        assert "[dry-run]" in result.stdout
        assert not config.exists()

    def test_uninstall_round_trip(self, tmp_hermes_home: Path) -> None:
        self._run()
        config = tmp_hermes_home / "config.yaml"
        self._run("--uninstall")
        after = config.read_text()
        # enabled list is empty, top-level provider blocks gone.
        assert "minimax-imggen" not in after
        assert "minimax-vidgen" not in after
        assert "minimax-musicgen" not in after
        assert "image_gen" not in after
        assert "video_gen" not in after
        assert "music_gen" not in after


# ---------------------------------------------------------------------------
# Plugin register() callbacks actually wire providers
# ---------------------------------------------------------------------------


class _StubContext:
    """Capture what each plugin registers so we can assert it."""

    def __init__(self) -> None:
        self.image_providers: list[Any] = []
        self.video_providers: list[Any] = []
        self.music_providers: list[Any] = []
        self.tools: list[tuple[str, Any]] = []
        self.hooks: list[tuple[str, Any]] = []

    def register_image_gen_provider(self, provider: Any) -> None:
        self.image_providers.append(provider)

    def register_video_gen_provider(self, provider: Any) -> None:
        self.video_providers.append(provider)

    def register_music_gen_provider(self, provider: Any) -> None:
        self.music_providers.append(provider)

    def register_tool(self, *args: Any, **kwargs: Any) -> None:
        self.tools.append((args, kwargs))  # type: ignore[arg-type]

    def register_hook(self, *args: Any, **kwargs: Any) -> None:
        self.hooks.append((args, kwargs))  # type: ignore[arg-type]


class TestPluginRegisterCallbacks:
    """Each plugin module's ``register(ctx)`` must call the right ctx method.

    Regression coverage for the moment a future refactor accidentally renames
    the ctx API or drops a provider.
    """

    def test_image_plugin_registers_image_provider(self) -> None:
        from hermes_minimax_media.plugins.image_gen import minimax as img

        ctx = _StubContext()
        img.register(ctx)
        assert len(ctx.image_providers) == 1
        assert ctx.image_providers[0].name == "minimax"

    def test_video_plugin_registers_video_provider(self) -> None:
        from hermes_minimax_media.plugins.video_gen import minimax as vid

        ctx = _StubContext()
        vid.register(ctx)
        assert len(ctx.video_providers) == 1
        assert ctx.video_providers[0].name == "minimax"

    def test_music_plugin_registers_something(self) -> None:
        """The music plugin's ``register()`` shape varies across versions —
        some versions register a provider, others register the tool directly
        because upstream Hermes' music_gen surface hasn't landed yet.
        Either way, ``register()`` must do *something* observable. """
        from hermes_minimax_media.plugins.music_gen import minimax as mus

        ctx = _StubContext()
        mus.register(ctx)
        registered = (
            len(ctx.music_providers) + len(ctx.tools) + len(ctx.hooks)
        )
        assert registered >= 1, (
            "music plugin register() did nothing — check that "
            "ctx.register_music_gen_provider or ctx.register_tool is called"
        )