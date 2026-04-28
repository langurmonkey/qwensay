#!/usr/bin/env python3
"""
Qwen3-TTS CLI — Generate speech with Voice Presets, Instructions, and Multilingual support.
"""

import argparse
import sys
import os
import subprocess
import shutil
from datetime import datetime

# ANSI Colors
class Color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def cprint(text, color=Color.END, bold=False):
    style = Color.BOLD if bold else ""
    print(f"{style}{color}{text}{Color.END}")

# Configuration
MODELS = {
    "1.7b": {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "description": "VoiceDesign — Create a new voice from description",
    },
    "1.7b-custom": {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "description": "CustomVoice — Use high-quality presets + style",
    },
    "0.6b": {
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "description": "0.6B CustomVoice — Lightweight and fast",
    },
}

# The 9 official Qwen3 presets
PRESETS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
LANGUAGES = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"]
DEFAULT_MODEL = "1.7b"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS CLI")
    parser.add_argument("--text", "-t", required=True, help="Text to speak.")
    parser.add_argument("--instruct", "-i", help="Style or voice description (e.g. 'whispering', 'excited').")
    parser.add_argument("--voice", "-v", choices=PRESETS, help="Voice preset name (CustomVoice models only).")
    parser.add_argument("--lang", "-l", default="English", choices=LANGUAGES, help="Language of the text.")
    parser.add_argument("--model", "-m", choices=list(MODELS.keys()), help="Model version.")
    parser.add_argument("--output", "-o", help="Output WAV path.")
    parser.add_argument("--device", default=None, help="cuda/cpu.")
    return parser.parse_args()

def play_audio(file_path):
    players = ["mpv", "play", "aplay", "vlc"]
    for player in players:
        if shutil.which(player):
            cmd = [player, file_path]
            if player == "mpv": cmd.append("--no-video")
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception: continue
    return False

def main():
    args = parse_args()
    
    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        cprint(f"Missing dependency: {e.name}", Color.RED)
        sys.exit(1)

    model_key = args.model or DEFAULT_MODEL
    model_id = MODELS[model_key]["id"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if "cuda" in device else torch.float32

    cprint(f"Qwen3-TTS | Model: {model_key} | Device: {device}", Color.BLUE, bold=True)
    cprint(f"Loading {model_id}...", Color.BLUE)
    
    model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)

    cprint(f"Generating ({args.lang})...", Color.GREEN)

    # Handle different model logic
    if "VoiceDesign" in model_id:
        # Instruction is the main driver here
        wavs, sr = model.generate_voice_design(
            text=args.text, 
            language=args.lang, 
            instruct=args.instruct or "A natural clear voice"
        )
    else:
        # CustomVoice requires a speaker preset
        speaker = args.voice or "Ryan"
        wavs, sr = model.generate_custom_voice(
            text=args.text, 
            language=args.lang, 
            speaker=speaker,
            instruct=args.instruct or ""
        )

    audio_data = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else wavs[0]
    out_path = args.output or f"outputs/qwen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    sf.write(out_path, audio_data, sr)

    if not args.output:
        cprint(f"Playing... (Saved to {out_path})", Color.GREEN)
        if not play_audio(out_path):
            cprint(f"No player found. File: {out_path}", Color.YELLOW)
    else:
        cprint(f"Saved: {out_path}", Color.GREEN)

if __name__ == "__main__":
    main()
