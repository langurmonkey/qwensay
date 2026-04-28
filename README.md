# QwenSay

A command-line tool for text-to-speech synthesis using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — supports free-form voice design, preset speakers, and voice cloning. Models are downloaded automatically from Hugging Face on first run.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- A CUDA-capable GPU (recommended; CPU works but is slow)
- [SoX](https://github.com/rbouqueau/SoX)
- [`mpv`](https://mpv.io) or `aplay` (optional, for automatic playback)

## Setup

```bash
# Clone or download the project, then enter the directory
cd qwensay

# Install default dependencies
uv sync
```

For Pascal (GTX 10x0 family), do this:

```bash
uv sync --group torch-pascal --no-group torch-default
```

On modern RTX cards, you want to use flash attention:

```bash
uv sync --extra gpu
```

> FlashAttention 2 is only used when the model is loaded in `bfloat16` on a CUDA device. The script enables it automatically if the package is present.

## Usage

```
uv run qwensay.py --text "Hello world" [OPTIONS]
```

### Required arguments

| Argument | Short | Description |
|---|---|---|
| `--text` | `-t` | The text to synthesize into speech |

### Optional arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--instruct` | `-i` | (none) | Natural-language description (e.g., "A raspy old man") or style (e.g., "Whispering") |
| `--voice` | `-v` | `Ryan` | Preset speaker name (used for `1.7b-custom` and `0.6b` models) |
| `--language` | `-l` | `English` | Language: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| `--output` | `-o` | `outputs/` | Save path. If omitted, plays audio and saves to outputs/ with a timestamp |
| `--model` | `-m` | *(interactive)* | `1.7b` (Voice Design), `1.7b-custom` (Presets), `0.6b` (Lightweight) |
| `--device` | | *(auto)* | PyTorch device, e.g. `cuda:0` or `cpu` |

### Choosing a model

If `--model` is omitted the script presents an interactive prompt. Pass `--model` to skip it.


| Key | Hugging Face ID | Best for | VRAM |
|---|---|---|---|
| `1.7b` *(default)* | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Free-form voice descriptions | ~6 GB |
| `1.7b-custom` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Preset speakers + style instructions | ~6 GB |
| `0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Voice cloning from reference audio | ~2 GB |

Models are downloaded to the Hugging Face cache (`~/.cache/huggingface/`) on first use and reused on subsequent runs.

#### Available preset speakers (`--voice`)

When using `1.7b-custom` or `0.6b`, you can choose from these official high-quality timbres:

- Female: `Vivian`, `Serena`, `Ono_Anna`, `Sohee`
- Male: `Ryan`, `Aiden`, `Uncle_Fu`, `Dylan`, `Eric`


## Examples

Voice design. Create a voice entirely from a description using the default 1.7B model.

```bash
uv run qwensay.py \
    --text "Good morning! Today is going to be a great day." \
    --voice "A cheerful, energetic young woman with a clear American accent"
```

Using a premium preset. Use a high-quality preset speaker with a specific emotional instruction.

```bash
uv run qwensay.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --voice "Deep, authoritative male narrator, slow and deliberate pace" \
    --model 1.7b
```

Multilingual. German text with a matching voice.
```bash
uv run qwensay.py \
    --text "Guten Morgen! Wie geht es Ihnen heute?" \
    --voice "Warm, professional male voice with a standard German accent" \
    --language German
```

## Notes

- **Output:** If no `--output` is specified, files are saved to the `./outputs/` folder with a timestamp and played automatically via `mpv` or `aplay`.
- **GTX 10-series Users:** Do not use FlashAttention 2. The script will automatically fall back to "eager" mode, which works perfectly on older Pascal cards.
- **VRAM:** If you experience "Out of Memory" errors on 6GB/8GB cards, ensure no other GPU-heavy apps are running, or switch to the `0.6b` model.
- The **VoiceDesign** model (`1.7b`) gives the most expressive results for arbitrary voice descriptions. Write descriptions in natural language: age, gender, accent, pace, and emotion all influence the output.
- The **0.6B Base** model is optimized for cloning a voice from a short reference WAV. Passing a `--instruct` description to it works on a best-effort basis.
- Output is always a 16-bit PCM WAV file at the sample rate returned by the model (typically 24 kHz).

## Alternatives

If you need a lighter model, you can try Kitten TTS through [puss-say](https://gihub.com/Mic92/puss-say). It runs in the CPU with minimal resources, and the configuration is straightforward. The speech quality is not too bad.

## License

Qwen3-TTS model weights are released under the [Apache 2.0 license](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE) by Alibaba Cloud / Qwen Team.

This project is licensed under GPL-3.0.
