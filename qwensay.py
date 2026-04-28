#!/usr/bin/env python3
"""
Qwen3-TTS CLI — Generate speech from text using a free-form voice description.
"""

import argparse
import sys
import os

# --- Available models ---
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
  python qwen3_tts.py --text "Hello world" --voice "A warm British female"
  python qwen3_tts.py -t "Narration text" -v "Deep male voice" -o audio/output.wav
        """
    )
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize.")
    parser.add_argument("--voice", "-v", required=True, help="Voice description or instruction.")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV path.")
    parser.add_argument("--model", "-m", choices=list(MODELS.keys()), help="Model version.")
    parser.add_argument("--language", "-l", default="English", help="Language hint.")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu).")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable FlashAttention 2.")
    return parser.parse_args()

def choose_model_interactively() -> str:
    print("\nAvailable Qwen3-TTS models:\n")
    keys = list(MODELS.keys())
    for i, k in enumerate(keys, 1):
        m = MODELS[k]
        print(f"  [{i}] {k:14s} {m['description']} (VRAM: {m['vram']})")
    
    print(f"\n  Press Enter for default ({DEFAULT_MODEL}).")
    while True:
        raw = input("Select model [1-3]: ").strip()
        if not raw: return DEFAULT_MODEL
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw)-1]
        print("Invalid selection.")

def main() -> None:
    args = parse_args()
    
    # Check for output directory
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}. Run: pip install qwen-tts soundfile torch", file=sys.stderr)
        sys.exit(1)

    model_key = args.model or choose_model_interactively()
    model_id = MODELS[model_key]["id"]
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if "cuda" in device else torch.float32

    # Safety check for Flash Attention
    use_flash = False
    if not args.no_flash_attn and "cuda" in device:
        try:
            import flash_attn
            use_flash = True
        except ImportError:
            use_flash = False
    attn_impl = "flash_attention_2" if use_flash else "eager"

    # Load model
    print(f"\n🔊 Qwen3-TTS | Model: {model_key} | Device: {device}")
    print("📥 Loading model...")
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_id, 
            device_map=device, 
            dtype=dtype, 
            attn_implementation=attn_impl
        )
    except Exception as e:
        if "flash_attention" in str(e).lower():
            print("⚠️ Flash Attention failed to load. Falling back to 'eager' implementation...")
            model = Qwen3TTSModel.from_pretrained(
                                        model_id,
                                        device_map=device,
                                        dtype=dtype,
                                        attn_implementation="eager")
        else:
            raise e

    print("🎙️ Generating...")
    
    # Mapping model type to the correct method
    if model_key == "1.7b":
        wavs, sr = model.generate_voice_design(text=args.text, language=args.language, instruct=args.voice)
    elif model_key == "1.7b-custom":
        wavs, sr = model.generate_custom_voice(text=args.text, language=args.language, instruct=args.voice)
    else:
        wavs, sr = model.generate_voice_clone(text=args.text, language=args.language, instruct=args.voice)

    # Ensure output is a CPU numpy array for soundfile
    audio_data = wavs[0]
    if hasattr(audio_data, "cpu"):
        audio_data = audio_data.cpu().numpy()

    sf.write(args.output, audio_data, sr)
    print(f"✅ Saved to: {os.path.abspath(args.output)} ({len(audio_data)/sr:.1f}s)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user.")
        sys.exit(0)
