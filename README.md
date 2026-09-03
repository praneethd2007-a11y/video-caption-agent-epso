# Quiptionary

Quiptionary is a video captioning agent built for the AMD Developer Hackathon (Track 2). Instead of generating the same flat description with a different adjective bolted on for "tone," each requested style is written as a distinct character with its own voice and attitude.

**[Live demo →](https://video-caption-agent-epso-ceyr9xaqesqqcgt2wsbp77.streamlit.app/)**

## The four characters

| Style | Character |
|---|---|
| `formal` | Cold, clinical precision, a HAL-9000-esque narrator |
| `sarcastic` | Weary, condescending observer, dry wit |
| `humorous_tech` | A burnt-out AI engineer narrating the world through bug reports |
| `humorous_non_tech` | An out-of-touch fifty-something bewildered by modern life |

## How it works

1. **Download** — fetch the clip from its URL
2. **Sample frames** - 4 evenly spaced frames pulled across the full clip (not just one snapshot), downscaled for faster inference
3. **Style prompt** - each style is a full persona prompt with strict output-format rules
4. **Vision model** - Qwen3.7 Plus (via Fireworks AI), reasoning/thinking mode disabled for speed and reliability
5. **Parse + verify** - extracts the tagged caption; any failure is clearly labeled rather than silently returned as broken text

Handles clips from 30 seconds to 2 minutes, well within the 10-minute container runtime limit.

## Run it

### Docker

```bash
docker pull ghcr.io/praneethd2007-a11y/video-caption-agent:latest

docker run --rm \
  -e FIREWORKS_API_KEY=your_key_here \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  ghcr.io/praneethd2007-a11y/video-caption-agent:latest
```

Reads tasks from `/input/tasks.json`, writes results to `/output/results.json`.

### Locally (no Docker)

```bash
pip install -r requirements.txt
python main.py
```

Reads from `practasks.json` / writes to `results.json` by default (override with `TASKS_PATH` / `RESULTS_PATH` env vars).

### Streamlit demo

```bash
streamlit run streamlit_app.py
```

## Tech stack

- **Fireworks AI** - hosted inference
- **Qwen3.7 Plus (vision)** - multi-image captioning
- **OpenCV** - frame extraction and downscaling
- **Docker** - `linux/amd64`, built and pushed via GitHub Actions to GHCR
- **Streamlit** - live demo UI

## Image

```
ghcr.io/praneethd2007-a11y/video-caption-agent:latest
```
