# Hermes MiniMax Media

[MiniMax](https://www.minimax.io) (海螺 / minimax) **image + video** generation backends for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

| Surface | Model | Modes |
|---|---|---|
| `image_gen/minimax` | `image-01` | Text-to-Image, Image-to-Image (subject reference) |
| `video_gen/minimax` | `MiniMax-Hailuo-2.3` | T2V, I2V, 6/10s @ 768P/1080P |
| `video_gen/minimax` | `MiniMax-Hailuo-02` | T2V, I2V with start-end frames |
| `video_gen/minimax` | `S2V-01` | Subject-reference T2V (face consistency) |

CN endpoint (`api.minimaxi.com`) is used by default when `MINIMAX_CN_API_KEY` is set;
global (`api.minimax.io`) is the automatic fallback when only `MINIMAX_API_KEY` is set.

## Demo

### Image (image-01)

| Apple on plate | Cat astronaut | Cyberpunk city |
|---|---|---|
| ![Apple](docs/assets/screenshot-apple.jpg) | ![Cat astronaut](docs/assets/screenshot-cat-astronaut.png) | ![Cyberpunk city](docs/assets/screenshot-city.png) |
| `a tiny red apple on a white plate, studio lighting, photorealistic` | `a cute cat astronaut floating in a colorful nebula, digital art, vibrant` | `futuristic tokyo cityscape at night, neon lights, rain reflections, cyberpunk` |

### Video (Hailuo-2.3, 6s @ 768P)

`a calico cat napping in a sunbeam, soft cinematic lighting, gentle breathing motion`

![Demo GIF](docs/assets/demo.gif)

[MP4 source (608KB)](docs/assets/screenshot-cat-napping.mp4)

## Installation

### From PyPI

```bash
pip install hermes-minimax-media
```

The package registers two entry points with the `hermes_agent.plugins` group
(`minimax-imggen` and `minimax-vidgen`). Hermes auto-discovers them on next
gateway start; you still need to enable the plugins in `config.yaml`
(see Configuration below).

### From source

```bash
git clone https://github.com/lora-sys/hermes-minimax-media.git
cd hermes-minimax-media
pip install -e .
```

### Manual install (no pip)

```bash
mkdir -p ~/.hermes/plugins/{image_gen,video_gen}
cp -r src/hermes_minimax_media/plugins/image_gen/minimax \
      ~/.hermes/plugins/image_gen/minimax
cp -r src/hermes_minimax_media/plugins/video_gen/minimax \
      ~/.hermes/plugins/video_gen/minimax
```

## Configuration

### 1. Set your MiniMax API key

Add to `~/.hermes/.env`:

```bash
# CN endpoint (api.minimaxi.com) — used by default
MINIMAX_CN_API_KEY=eyJ...

# Optional: global endpoint (api.minimax.io) — fallback
MINIMAX_API_KEY=eyJ...
```

Get a key at https://api.minimaxi.com/user-center/basic-information/interface-key
(or https://www.minimax.io for the global endpoint).

### 2. Enable the plugins in `~/.hermes/config.yaml`

```yaml
plugins:
  enabled:
    - image_gen/minimax
    - video_gen/minimax
    # ... other plugins

image_gen:
  provider: minimax
  minimax:
    model: image-01

video_gen:
  provider: minimax
  minimax:
    model: MiniMax-Hailuo-2.3   # or Hailuo-02, S2V-01
```

### 3. Restart Hermes

```bash
hermes gateway restart
```

## Usage

Once configured, ask Hermes naturally — the model picks the right tool:

**Text-to-image:**
> 画一只戴墨镜的猫

**Image-to-image** (subject reference):
> 把这张自拍转成吉卜力风格 (pass an `image_url`)

**Text-to-video:**
> 做一个 6 秒的延时摄影：城市黄昏的车流

**Image-to-video** (animate a still):
> 让这张海浪照片动起来 (Hailuo-2.3, pass `image_url`)

**Subject-reference video** (face-consistent character):
> 让视频里这个角色挥手 (S2V-01, pass `image_url` of a face)

## Models

### Image

| Model | Modes | Aspect ratios | Notes |
|---|---|---|---|
| `image-01` | T2I, I2I | 16:9, 9:16, 1:1 | Built-in prompt optimizer on; `seed` supported |

### Video

| Model | Duration | Resolution | T2V | I2V | Subject ref | Start-end frames |
|---|---|---|---|---|---|---|
| `MiniMax-Hailuo-2.3` | 6 / 10 s | 768P, 1080P | ✅ | ✅ | ❌ | ❌ |
| `MiniMax-Hailuo-02` | 6 / 10 s | 768P, 1080P | ✅ | ✅ | ❌ | ✅ |
| `S2V-01` | 6 s | 1080P | ❌ | ❌ | ✅ (face) | ❌ |

> Note: `720P` is **not** a valid MiniMax resolution. The plugin auto-coerces
> `720P` -> `768P` to avoid API errors. Don't rely on `720P` working.

## Endpoint resolution

| Env state | Endpoint used |
|---|---|
| `MINIMAX_CN_API_KEY` set | `https://api.minimaxi.com` (CN, default) |
| Only `MINIMAX_API_KEY` set | `https://api.minimax.io` (global) |
| `MINIMAX_CN_BASE_URL` / `MINIMAX_BASE_URL` set | Override host only; paths are always `/v1/...` |

## Troubleshooting

### `MiniMax image API error 401`

Wrong key or the wrong region. Verify the key is valid at the relevant user
center (CN: https://api.minimaxi.com ; global: https://www.minimax.io). The
plugin auto-picks the right endpoint based on which env var is set, but if
both are set, CN wins.

### `MiniMax submit returned no task_id: ... does not support resolution 720P`

`720P` was a wrong assumption from older docs. The plugin normalizes it to
`768P`. If you see this error from a hand-rolled request, switch to `768P`.

### Video generation times out (>5 min)

The plugin polls every 10s for up to 5 minutes. Hailuo-2.3 typically returns
in 30-90s for a 6s/768P clip; 10s/1080P can stretch to 3-4 minutes. If you
hit the 5-min cap repeatedly, switch to a shorter duration or lower
resolution.

### `MiniMax submit error 429: ... balance not enough`

Out of credits. Top up at https://api.minimaxi.com/user-center/payment
(or the global user center).

### Plugin not loading

Verify the path matches what `config.yaml` says:

```bash
ls ~/.hermes/plugins/image_gen/minimax/
ls ~/.hermes/plugins/video_gen/minimax/
```

Each directory should contain both `__init__.py` and `plugin.yaml`. If you
installed via `pip install hermes-minimax-media`, the entry points
`minimax-imggen` and `minimax-vidgen` are auto-registered; the
`image_gen/minimax` and `video_gen/minimax` names still need to be added
to `plugins.enabled`.

## Development

```bash
# Clone
git clone https://github.com/lora-sys/hermes-minimax-media
cd hermes-minimax-media

# Editable install
pip install -e .

# Run tests (no API key needed)
pytest tests/ -v

# Run live smoke (requires MINIMAX_CN_API_KEY)
MINIMAX_CN_API_KEY=... pytest tests/ -v -m requires_live

# Lint
ruff check src/ tests/

# Build
python -m build
```

## License

MIT
