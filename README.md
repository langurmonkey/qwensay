# QwenSay

A command-line tool for text-to-speech synthesis using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — supports free-form voice design, preset speakers, and voice cloning. Models are downloaded automatically from Hugging Face on first run.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A CUDA-capable GPU (recommended; CPU works but is slow)
- [SoX](https://github.com/rbouqueau/SoX)

## Setup

```bash
# Clone or download the project, then enter the directory
cd qwensay

# Install all dependencies from the lockfile
uv sync
```

**Optional — FlashAttention 2** (reduces GPU memory usage, requires a compatible NVIDIA GPU):

```bash
uv sync --extra gpu
```

> FlashAttention 2 is only used when the model is loaded in `bfloat16` on a CUDA device. The script enables it automatically if the package is present.

## Usage

```
uv run qwensay.py --text TEXT --voice VOICE [options]
```

### Required arguments

| Argument | Short | Description |
|---|---|---|
| `--text` | `-t` | The text to synthesize into speech |
| `--voice` | `-v` | Natural-language description of the desired voice |

### Optional arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--output` | `-o` | `output.wav` | Path for the generated `wav` file |
| `--model` | `-m` | *(interactive)* | Model to use — see table below |
| `--language` | `-l` | `English` | Language hint for the model |
| `--device` | | *(auto)* | PyTorch device, e.g. `cuda:0` or `cpu` |
| `--no-flash-attn` | | | Disable FlashAttention 2 even if available |

### Choosing a model

If `--model` is omitted the script presents an interactive prompt. Pass `--model` to skip it.

| Key | Hugging Face ID | Best for | VRAM |
|---|---|---|---|
| `1.7b` *(default)* | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Free-form voice descriptions | ~6 GB |
| `1.7b-custom` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Preset speakers + style instructions | ~6 GB |
| `0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Voice cloning from reference audio | ~2 GB |

Models are downloaded to the Hugging Face cache (`~/.cache/huggingface/`) on first use and reused on subsequent runs.

### Supported languages

English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

## Examples

```bash
# Basic — model is selected interactively on first run
uv run qwensay.py \
    --text "Good morning! Today is going to be a great day." \
    --voice "A cheerful, energetic young woman with a clear American accent"

# Specify model and output path explicitly
uv run qwensay.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --voice "Deep, authoritative male narrator, slow and deliberate pace" \
    --model 1.7b \
    --output narration.wav

# Multilingual — German text with a matching voice
uv run qwensay.py \
    --text "Guten Morgen! Wie geht es Ihnen heute?" \
    --voice "Warm, professional male voice with a standard German accent" \
    --language German \
    --output german.wav

# Lightweight model on CPU (slower, no GPU needed)
uv run qwensay.py \
    --text "Hello from CPU." \
    --voice "Calm, neutral female voice" \
    --model 0.6b \
    --device cpu
```

## Notes

- The **VoiceDesign** model (`1.7b`) gives the most expressive results for arbitrary voice descriptions. Write descriptions in natural language — age, gender, accent, pace, and emotion all influence the output.
- The **0.6B Base** model is optimised for cloning a voice from a short reference WAV. Passing a `--voice` description to it works on a best-effort basis.
- Output is always a 16-bit PCM WAV file at the sample rate returned by the model (typically 24 kHz).

## License

Qwen3-TTS model weights are released under the [Apache 2.0 license](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE) by Alibaba Cloud / Qwen Team.
