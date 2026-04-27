#!/usr/bin/env python3
"""
Qwen3-TTS CLI — Generate speech from text using a free-form voice description.

Usage:
    python qwen3_tts.py --text "Hello, world!" --voice "A warm, friendly female voice with a slight British accent"
    python qwen3_tts.py --text "..." --voice "..." --output my_audio.wav --model 1.7b

Requirements:
    pip install qwen-tts soundfile
    # Optional but recommended (needs compatible GPU):
    pip install flash-attn --no-build-isolation
"""

import argparse
import sys
import os

# ── Available models ──────────────────────────────────────────────────────────
MODELS = {
    "1.7b": {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "description": "1.7B VoiceDesign — best quality, free-form voice descriptions (recommended)",
        "vram": "~6 GB",
    },
    "1.7b-custom": {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "description": "1.7B CustomVoice — preset speakers + style instructions",
        "vram": "~6 GB",
    },
    "0.6b": {
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "description": "0.6B Base — fastest / lowest VRAM, voice-clone only",
        "vram": "~2 GB",
    },
}

DEFAULT_MODEL = "1.7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: text-to-speech with free-form voice design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic (auto-downloads 1.7B VoiceDesign model on first run):
  python qwen3_tts.py \\
      --text "Good morning! Today is going to be a great day." \\
      --voice "A cheerful, energetic young woman with a clear American accent"

  # Save to a custom path:
  python qwen3_tts.py \\
      --text "The quick brown fox." \\
      --voice "Deep, authoritative male narrator, slow pace" \\
      --output narration.wav

  # Use the lighter 0.6B model:
  python qwen3_tts.py \\
      --text "Hola mundo." \\
      --voice "Warm female voice, Spanish accent" \\
      --model 0.6b

Available models:
""" + "\n".join(f"  {k:14s}  {v['description']}  (VRAM: {v['vram']})" for k, v in MODELS.items()),
    )
    parser.add_argument(
        "--text", "-t",
        required=True,
        help="The text to synthesize into speech.",
    )
    parser.add_argument(
        "--voice", "-v",
        required=True,
        help=(
            "Natural-language description of the desired voice. "
            "E.g. 'A calm, deep male voice with a British accent'."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default="output.wav",
        help="Path for the generated WAV file (default: output.wav).",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default=None,
        help=(
            f"Which Qwen3-TTS model to use. "
            f"Default: {DEFAULT_MODEL} (prompted interactively if omitted)."
        ),
    )
    parser.add_argument(
        "--language", "-l",
        default="English",
        help=(
            "Language hint for the model. "
            "Supported: English, Chinese, Japanese, Korean, German, French, "
            "Russian, Portuguese, Spanish, Italian. (default: English)"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "PyTorch device string, e.g. 'cuda:0' or 'cpu'. "
            "Auto-detected if not set."
        ),
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable FlashAttention 2 even if available.",
    )
    return parser.parse_args()


def choose_model_interactively() -> str:
    """Ask the user which model to use when --model is not provided."""
    print("\nAvailable Qwen3-TTS models:\n")
    keys = list(MODELS.keys())
    for i, k in enumerate(keys, 1):
        m = MODELS[k]
        print(f"  [{i}] {k:14s}  {m['description']}")
        print(f"       HF model ID : {m['id']}")
        print(f"       GPU VRAM    : {m['vram']}")
        print()

    print(f"  Press Enter to use the default ({DEFAULT_MODEL}).")
    while True:
        raw = input("Select model [1/2/3 or name]: ").strip()
        if raw == "":
            return DEFAULT_MODEL
        # Accept number
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        # Accept name directly
        if raw in MODELS:
            return raw
        print(f"  ⚠  Invalid choice '{raw}'. Please enter a number (1–{len(keys)}) or a model name.")


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_dependencies() -> None:
    """Check that required packages are importable, give a clear message if not."""
    missing = []
    for pkg in ("qwen_tts", "soundfile", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            "\n❌  Missing required package(s): "
            + ", ".join(missing)
            + "\n\nInstall with:\n"
            "  pip install qwen-tts soundfile\n\n"
            "For GPU acceleration (optional):\n"
            "  pip install flash-attn --no-build-isolation\n",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    args = parse_args()

    # ── Dependency check ──────────────────────────────────────────────────────
    ensure_dependencies()

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    # ── Model selection ───────────────────────────────────────────────────────
    model_key = args.model if args.model else choose_model_interactively()
    model_info = MODELS[model_key]
    model_id = model_info["id"]

    # ── Device & dtype ────────────────────────────────────────────────────────
    device = args.device or detect_device()
    dtype = torch.bfloat16 if "cuda" in device else torch.float32

    use_flash = (
        not args.no_flash_attn
        and has_flash_attn()
        and dtype in (torch.float16, torch.bfloat16)
        and "cuda" in device
    )
    attn_impl = "flash_attention_2" if use_flash else "eager"

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n🔊  Qwen3-TTS")
    print(f"    Model   : {model_id}")
    print(f"    Device  : {device}  |  dtype: {dtype}  |  FlashAttn: {use_flash}")
    print(f"    Voice   : {args.voice!r}")
    print(f"    Language: {args.language}")
    print(f"    Text    : {args.text[:80]}{'…' if len(args.text) > 80 else ''}")
    print(f"    Output  : {os.path.abspath(args.output)}\n")

    # ── Load model (HF cache handles download automatically) ──────────────────
    print("📥  Loading model (downloads on first run — may take a few minutes)…")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    print("✅  Model loaded.\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    print("🎙️  Generating speech…")

    if model_key == "1.7b":
        # VoiceDesign: free-form voice description
        wavs, sr = model.generate_voice_design(
            text=args.text,
            language=args.language,
            instruct=args.voice,
        )
    elif model_key == "1.7b-custom":
        # CustomVoice: the voice description becomes the instruct field;
        # no fixed speaker is forced so the model follows the description.
        wavs, sr = model.generate_custom_voice(
            text=args.text,
            language=args.language,
            instruct=args.voice,
        )
    else:
        # 0.6B Base: voice-clone model; voice description used as instruct
        # (best effort — Base model primarily supports cloning).
        print(
            "⚠  The 0.6B Base model is optimised for voice cloning (needs a reference WAV).\n"
            "   The --voice description will be passed as an instruction but results may vary.\n"
            "   For free-form voice design use --model 1.7b.\n"
        )
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            instruct=args.voice,
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    sf.write(args.output, wavs[0], sr)
    print(f"\n✅  Audio saved → {os.path.abspath(args.output)}")
    duration = len(wavs[0]) / sr
    print(f"    Duration: {duration:.1f} s  |  Sample rate: {sr} Hz\n")


if __name__ == "__main__":
    main()
