"""One-shot setup helper for the hermes-minimax-media package.

After `pip install hermes-minimax-media`, this script adds the three MiniMax
plugin entry points (``minimax-imggen``, ``minimax-vidgen``, ``minimax-musicgen``)
to ``~/.hermes/config.yaml``'s ``plugins.enabled`` list and writes sensible
defaults for the ``image_gen``, ``video_gen``, and ``music_gen`` provider
blocks. Idempotent — re-running on an already-configured file is a no-op.

Usage::

    hermes-minimax-setup                # enable + write defaults
    hermes-minimax-setup --check        # exit 0 if configured, 1 otherwise
    hermes-minimax-setup --uninstall    # remove from plugins.enabled + provider blocks
    hermes-minimax-setup --print        # show what would change (dry-run)
    hermes-minimax-setup --json         # machine-readable status

Why this exists
---------------
Hermes' plugin loader (``hermes_cli/plugins.py:_scan_entry_points``) creates
plugin manifests from entry points with ``key=ep.name`` (e.g.
``"minimax-vidgen"``) and ``kind="standalone"`` — it does NOT read the
bundled ``plugin.yaml``. So writing ``plugins.enabled: [video_gen/minimax]``
(the path-derived key, what the bundled layout would emit) does NOT match an
entry-point-installed plugin. The opt-in allow-list needs the **entry-point
name** itself (``minimax-vidgen``). This script does that write so users don't
have to remember the exact spelling or open ``config.yaml`` by hand.

If/when upstream Hermes' entry-point loader learns to read the yaml kind
(see the "kind: backend" auto-detect that the bundled-layout scanner
already does), this script becomes a one-shot migration aid rather than
a permanent install step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Entry point names (must match pyproject.toml)
ENTRY_POINTS = ("minimax-imggen", "minimax-vidgen", "minimax-musicgen")

# Provider blocks we want to set up. Format: (provider-block, model, entry-point)
PROVIDER_BLOCKS = {
    "image_gen": {
        "provider": "minimax",
        "minimax": {"model": "image-01"},
    },
    "video_gen": {
        "provider": "minimax",
        "minimax": {"model": "MiniMax-Hailuo-2.3"},
    },
    "music_gen": {
        "minimax": {"model": "music-2.6"},
    },
}


def _config_path() -> Path:
    """Locate ``~/.hermes/config.yaml`` (env override ``HERMES_HOME``)."""
    home = os.environ.get("HERMES_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "config.yaml"
    return Path.home() / ".hermes" / "config.yaml"


def _yaml_module():
    """Import PyYAML lazily. The package isn't a runtime dep of the plugin
    modules themselves; only this setup helper needs it. We keep it optional
    so `pip install hermes-minimax-media` doesn't pull yaml for users who
    never run the setup helper.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        sys.exit(
            "PyYAML is required for hermes-minimax-setup.\n"
            "Install with:  pip install 'hermes-minimax-media[setup]'  "
            "or  pip install pyyaml\n"
            f"({exc})"
        )
    return yaml


def load_config(path: Path) -> dict[str, Any]:
    yaml = _yaml_module()
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        sys.exit(f"Could not parse {path}: {exc}")


def save_config(path: Path, cfg: dict[str, Any]) -> None:
    yaml = _yaml_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


def current_status(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable description of the current setup state."""
    plugins_raw = cfg.get("plugins")
    plugins: dict[str, Any] = plugins_raw if isinstance(plugins_raw, dict) else {}
    enabled_raw = plugins.get("enabled")
    enabled: set[str] = set(enabled_raw) if isinstance(enabled_raw, list) else set()

    image_block_raw = cfg.get("image_gen")
    image_block: dict[str, Any] = image_block_raw if isinstance(image_block_raw, dict) else {}
    video_block_raw = cfg.get("video_gen")
    video_block: dict[str, Any] = video_block_raw if isinstance(video_block_raw, dict) else {}
    music_block_raw = cfg.get("music_gen")
    music_block: dict[str, Any] = music_block_raw if isinstance(music_block_raw, dict) else {}

    music_minimax_raw = music_block.get("minimax")
    music_minimax: dict[str, Any] = music_minimax_raw if isinstance(music_minimax_raw, dict) else {}

    return {
        "config_path": str(_config_path()),
        "plugins_enabled": sorted(enabled),
        "missing_entry_points": sorted(set(ENTRY_POINTS) - enabled),
        "image_gen_provider": image_block.get("provider"),
        "video_gen_provider": video_block.get("provider"),
        "music_gen_minimax_model": music_minimax.get("model"),
        "fully_configured": all(ep in enabled for ep in ENTRY_POINTS),
    }


def apply_setup(cfg: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Mutate ``cfg`` in place to enable + configure everything; return the diff."""
    diff: dict[str, Any] = {"added_to_enabled": [], "wrote_provider_blocks": []}

    plugins = cfg.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        cfg["plugins"] = plugins
    enabled_list = plugins.setdefault("enabled", [])
    if not isinstance(enabled_list, list):
        enabled_list = []
        plugins["enabled"] = enabled_list
    enabled_set = set(enabled_list)
    for ep in ENTRY_POINTS:
        if ep not in enabled_set:
            enabled_list.append(ep)
            enabled_set.add(ep)
            diff["added_to_enabled"].append(ep)

    for block_name, block_value in PROVIDER_BLOCKS.items():
        # image_gen / video_gen / music_gen each have a provider + per-provider
        # model block. music_gen's provider top-level is omitted in some
        # configs (it's a fallback surface); only set the per-provider block.
        if block_name in ("image_gen", "video_gen"):
            current = cfg.get(block_name) or {}
            if not isinstance(current, dict):
                current = {}
            changed = False
            # Don't clobber an existing provider choice (e.g. user already
            # set image_gen.provider = openai). We still write the per-
            # provider minimax block so the model is pre-configured if they
            # switch later, and so the plugin knows its default model.
            existing_provider = current.get("provider")
            if existing_provider is None:
                current["provider"] = block_value["provider"]
                changed = True
            for prov, sub in block_value.items():
                if prov == "provider":
                    continue
                cur_sub = current.get(prov) or {}
                if not isinstance(cur_sub, dict):
                    cur_sub = {}
                for k, v in sub.items():
                    if cur_sub.get(k) != v:
                        cur_sub[k] = v
                        changed = True
                current[prov] = cur_sub
            if changed:
                diff["wrote_provider_blocks"].append(block_name)
            cfg[block_name] = current
        elif block_name == "music_gen":
            current = cfg.get("music_gen") or {}
            if not isinstance(current, dict):
                current = {}
            # Don't set top-level music_gen.provider (upstream surface is
            # still landing — see README). Just set the minimax model.
            cur_minimax = current.get("minimax") or {}
            if not isinstance(cur_minimax, dict):
                cur_minimax = {}
            changed = False
            for k, v in block_value["minimax"].items():
                if cur_minimax.get(k) != v:
                    cur_minimax[k] = v
                    changed = True
            if changed:
                diff["wrote_provider_blocks"].append("music_gen")
            current["minimax"] = cur_minimax
            cfg["music_gen"] = current

    return diff


def apply_uninstall(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reverse of apply_setup — remove entries and clear provider models
    we wrote. Leaves non-MiniMax provider blocks untouched (e.g. user may
    have set image_gen.provider to openai separately)."""
    diff: dict[str, list[str]] = {
        "removed_from_enabled": [],
        "cleared_provider_blocks": [],
    }
    plugins_raw = cfg.get("plugins")
    plugins: dict[str, Any] = plugins_raw if isinstance(plugins_raw, dict) else {}
    enabled_raw = plugins.get("enabled")
    enabled: list[Any] = enabled_raw if isinstance(enabled_raw, list) else []
    ep_set = set(ENTRY_POINTS)
    new_enabled = [e for e in enabled if e not in ep_set]
    diff["removed_from_enabled"] = [e for e in enabled if e in ep_set]
    plugins["enabled"] = new_enabled
    cfg["plugins"] = plugins

    for block_name in ("image_gen", "video_gen"):
        current = cfg.get(block_name)
        if isinstance(current, dict) and current.get("provider") == "minimax":
            current.pop("provider", None)
            current.pop("minimax", None)
            cfg[block_name] = current
            diff["cleared_provider_blocks"].append(block_name)
    # music_gen: only the minimax sub-block
    mg = cfg.get("music_gen")
    if isinstance(mg, dict) and "minimax" in mg:
        mg.pop("minimax", None)
        cfg["music_gen"] = mg
        diff["cleared_provider_blocks"].append("music_gen")

    # Tidy: drop any top-level blocks we emptied out, so uninstall leaves the
    # config identical to its pre-install state.
    for block_name in ("image_gen", "video_gen", "music_gen"):
        v = cfg.get(block_name)
        if isinstance(v, dict) and not v:
            cfg.pop(block_name, None)

    return diff


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hermes-minimax-setup",
        description="Configure ~/.hermes/config.yaml to load the MiniMax plugins.",
    )
    ap.add_argument("--check", action="store_true",
                    help="exit 0 if fully configured, 1 otherwise (no writes)")
    ap.add_argument("--uninstall", action="store_true",
                    help="reverse the setup (remove entries + clear provider blocks)")
    ap.add_argument("--print", dest="dry_run", action="store_true",
                    help="dry run: show what would change, don't write")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON to stdout")
    args = ap.parse_args(argv)

    path = _config_path()
    cfg = load_config(path)

    if args.check:
        status = current_status(cfg)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            if status["fully_configured"]:
                print(f"OK  {path}")
                print(f"    enabled: {', '.join(status['plugins_enabled']) or '(none)'}")
                return 0
            print(f"NOT CONFIGURED  {path}")
            if status["missing_entry_points"]:
                print(f"    missing: {', '.join(status['missing_entry_points'])}")
            return 1

    if args.uninstall:
        diff = apply_uninstall(cfg)
        if not args.dry_run:
            save_config(path, cfg)
        if args.json:
            print(json.dumps({"action": "uninstall", "path": str(path),
                              "diff": diff, "dry_run": args.dry_run}, indent=2))
        else:
            print(f"{'[dry-run] ' if args.dry_run else ''}Updated {path}")
            if diff["removed_from_enabled"]:
                print(f"  removed from plugins.enabled: "
                      f"{', '.join(diff['removed_from_enabled'])}")
            if diff["cleared_provider_blocks"]:
                print(f"  cleared provider blocks: "
                      f"{', '.join(diff['cleared_provider_blocks'])}")
        return 0

    # Default action: apply setup.
    diff = apply_setup(cfg, dry_run=args.dry_run)
    if not args.dry_run:
        save_config(path, cfg)
    if args.json:
        print(json.dumps({"action": "setup", "path": str(path),
                          "diff": diff, "dry_run": args.dry_run,
                          "status": current_status(cfg)}, indent=2))
    else:
        print(f"{'[dry-run] ' if args.dry_run else ''}Configured {path}")
        if diff["added_to_enabled"]:
            print(f"  added to plugins.enabled: "
                  f"{', '.join(diff['added_to_enabled'])}")
        if diff["wrote_provider_blocks"]:
            print(f"  wrote provider blocks: "
                  f"{', '.join(diff['wrote_provider_blocks'])}")
        if not diff["added_to_enabled"] and not diff["wrote_provider_blocks"]:
            print("  no changes — already configured")
    return 0


if __name__ == "__main__":
    sys.exit(main())