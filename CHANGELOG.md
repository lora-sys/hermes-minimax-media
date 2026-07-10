# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-10

### Added
- Image generation backend (`image_gen/minimax`): MiniMax `image-01` model
  - Text-to-image (T2I) with aspect_ratio 16:9 / 9:16 / 1:1
  - Image-to-image (I2I) via `subject_reference` (single reference image)
  - CN endpoint (`api.minimaxi.com`) first, global (`api.minimax.io`) fallback
  - Auto-resolves `MINIMAX_CN_API_KEY` -> `MINIMAX_API_KEY`
  - Optional `prompt_optimizer` and `seed` kwargs passthrough
- Video generation backend (`video_gen/minimax`): MiniMax Hailuo family
  - `MiniMax-Hailuo-2.3` (T2V + I2V, 6s/10s, 768P/1080P)
  - `MiniMax-Hailuo-02` (T2V + I2V with start-end frames, 6s/10s, 768P/1080P)
  - `S2V-01` (subject-reference T2V, 6s, 1080P)
  - Async task submission -> polled status -> file retrieve -> local cache
  - 5-minute total budget, 10s poll cadence
- Standard pip package structure with two Hermes entry points
- Pytest unit test suite (no network required) with optional live smoke marker
- GitHub Actions CI (test on py3.11/3.12 + ruff lint + auto PyPI release on tag)
- GitHub repo: https://github.com/lora-sys/hermes-minimax-media
- PyPI: https://pypi.org/project/hermes-minimax-media/

### Notes
- `720P` is intentionally not advertised; the MiniMax API returns
  "MiniMax-Hailuo-2.3 does not support resolution 720P, supported resolutions: 768P, 1080P"
  for that value. The plugin normalizes `720P` -> `768P` automatically.
