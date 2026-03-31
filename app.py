import os
import sys
import shutil
import subprocess
import threading
import time
import re
import json
import warnings
import wave
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
except ImportError:
    print("Error: 'mlx_audio' library not found.")
    print("Run: source .venv/bin/activate")
    sys.exit(1)

import gradio as gr

from main import (
    MODELS,
    SPEAKER_CHOICES,
    EMOTION_EXAMPLES,
    VOICES_DIR,
    BASE_OUTPUT_DIR,
    get_smart_path,
    get_saved_voices,
    convert_audio_if_needed,
    clean_memory,
)

from datetime import datetime

_model_cache: dict = {}

PRESETS_DIR = os.path.join(os.getcwd(), "presets")
APP_SETTINGS_PATH = os.path.join(PRESETS_DIR, "app_settings.json")

MODE_TO_KEY = {"custom": "1", "design": "2", "clone": "3"}

SPEED_MAP = {"Normal (1.0x)": 1.0, "Fast (1.3x)": 1.3, "Slow (0.8x)": 0.8}

# Labels → lang_code for mlx_audio.generate_audio. Qwen3-TTS officially documents 10
# languages (see https://qwenlm-qwen3-tts.mintlify.app/concepts/languages); Thai and
# some other codes are passed through for convenience — use Auto or a documented
# language when quality matters.
LANG_CHOICES = [
    ("Auto", "auto"),
    ("English", "english"),
    ("Chinese", "chinese"),
    ("Japanese", "japanese"),
    ("Korean", "korean"),
    ("French", "french"),
    ("German", "german"),
    ("Spanish", "spanish"),
    ("Portuguese", "portuguese"),
    ("Italian", "italian"),
    ("Arabic", "arabic"),
    ("Russian", "russian"),
    ("Dutch", "dutch"),
    ("Thai", "thai"),
]

SPEAKER_LABELS = [label for label, _ in SPEAKER_CHOICES]
SPEAKER_ID_BY_LABEL = {label: sid for label, sid in SPEAKER_CHOICES}

# Character profile "Base Speaker" dropdown: explicit none so optional field can be cleared
BASE_SPEAKER_NONE = "(none)"
BASE_SPEAKER_DROPDOWN_CHOICES = [BASE_SPEAKER_NONE] + SPEAKER_LABELS


def _speaker_id_from_ui(value: str) -> str:
    """Map dropdown label (or legacy preset id) to lowercase API voice id."""
    if value in SPEAKER_ID_BY_LABEL:
        return SPEAKER_ID_BY_LABEL[value]
    low = str(value).strip().lower().replace(" ", "_")
    for _, sid in SPEAKER_CHOICES:
        if sid == low:
            return sid
    return low


def _label_for_speaker_id(sid: str) -> str:
    sid = (sid or "vivian").strip().lower().replace(" ", "_")
    for label, sid2 in SPEAKER_CHOICES:
        if sid2 == sid:
            return label
    return SPEAKER_CHOICES[0][0]


def _text_for_model(text: str, remove_linebreaks: bool) -> str:
    """If remove_linebreaks: collapse whitespace (newlines → single spaces)."""
    if not remove_linebreaks:
        return text
    return " ".join(text.split())


def _segments_for_history(text: str) -> list[str]:
    """Split text into non-empty lines (used for Separate by line)."""
    if not text:
        return []
    return [ln.strip() for ln in str(text).splitlines() if ln.strip()]


DEFAULT_APP_SETTINGS = {
    "lang": "auto",
    "temperature": 0.7,
    "autoplay": False,
    "remove_linebreaks": False,
    "separate_by_line": False,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))


def load_app_settings() -> dict:
    """Load persisted UI settings. Returns a validated dict."""
    allowed_langs = {v for _, v in LANG_CHOICES}
    data = {}
    try:
        if os.path.exists(APP_SETTINGS_PATH):
            with open(APP_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
    except Exception:
        data = {}

    lang = data.get("lang", DEFAULT_APP_SETTINGS["lang"])
    if lang not in allowed_langs:
        lang = DEFAULT_APP_SETTINGS["lang"]

    temp = _clamp(data.get("temperature", DEFAULT_APP_SETTINGS["temperature"]), 0.1, 1.5)
    ap = bool(data.get("autoplay", DEFAULT_APP_SETTINGS["autoplay"]))
    nlb = bool(data.get("remove_linebreaks", DEFAULT_APP_SETTINGS["remove_linebreaks"]))
    sbl = bool(data.get("separate_by_line", DEFAULT_APP_SETTINGS["separate_by_line"]))

    if nlb and sbl:
        sbl = False

    return {
        "lang": lang,
        "temperature": temp,
        "autoplay": ap,
        "remove_linebreaks": nlb,
        "separate_by_line": sbl,
    }


def save_app_settings(settings: dict) -> None:
    os.makedirs(PRESETS_DIR, exist_ok=True)
    with open(APP_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


CSS = """
.global-settings-row {
  align-items: center !important;
}
"""


def _play_audio(path: str):
    """Play audio via macOS afplay in a background thread so generation isn't blocked."""
    try:
        subprocess.run(
            ["afplay", path], check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def _maybe_play(path: str, autoplay: bool):
    if autoplay and path:
        threading.Thread(target=_play_audio, args=(path,), daemon=True).start()


# ── Model loading ─────────────────────────────────────────────────────────────

def get_model(model_key: str):
    global _model_cache
    if model_key not in _model_cache:
        _model_cache.clear()
        clean_memory()
        model_path = get_smart_path(MODELS[model_key]["folder"])
        if not model_path:
            raise RuntimeError(
                f"Model folder not found: {MODELS[model_key]['folder']}\n"
                "Download it from HuggingFace and place it in the models/ directory."
            )
        _model_cache[model_key] = load_model(model_path)
    return _model_cache[model_key]


# ── File helpers ──────────────────────────────────────────────────────────────

def _save_output(temp_dir: str, subfolder: str, text: str) -> str:
    save_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    clean_text = (
        re.sub(r"[^\w\s-]", "", text)[:20].strip().replace(" ", "_") or "audio"
    )
    base = os.path.join(save_path, f"{timestamp}_{clean_text}")
    final_path = f"{base}.wav"
    i = 1
    while os.path.exists(final_path):
        final_path = f"{base}_{i}.wav"
        i += 1

    source = os.path.join(temp_dir, "audio_000.wav")
    if os.path.exists(source):
        shutil.move(source, final_path)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return final_path if os.path.exists(final_path) else None


def _save_all_outputs(temp_dir: str, subfolder: str, text_lines: list[str]) -> list[str]:
    """
    Save all generated audio clips from a temp folder (audio_000.wav, audio_001.wav, ...)
    into the outputs/<subfolder>/ directory.
    """
    save_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(save_path, exist_ok=True)

    out_paths: list[str] = []
    try:
        if not os.path.isdir(temp_dir):
            return []

        wavs = [
            f for f in os.listdir(temp_dir)
            if re.fullmatch(r"audio_\d{3}\.wav", f)
        ]
        wavs.sort()

        timestamp = datetime.now().strftime("%H-%M-%S")
        for idx, fname in enumerate(wavs):
            source = os.path.join(temp_dir, fname)
            snippet = text_lines[idx] if idx < len(text_lines) else (text_lines[0] if text_lines else "audio")
            clean_text = (
                re.sub(r"[^\w\s-]", "", snippet)[:20].strip().replace(" ", "_") or "audio"
            )
            base = os.path.join(save_path, f"{timestamp}_{clean_text}_{idx+1:02d}")
            final_path = f"{base}.wav"
            n = 1
            while os.path.exists(final_path):
                final_path = f"{base}_{n}.wav"
                n += 1
            shutil.move(source, final_path)
            if os.path.exists(final_path):
                out_paths.append(final_path)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return out_paths


# ── Preset helpers ────────────────────────────────────────────────────────────

def _preset_path(category: str) -> str:
    return os.path.join(PRESETS_DIR, category)


# ── Character Management ──────────────────────────────────────────────────────

def get_characters() -> list:
    """Get list of saved character names."""
    path = _preset_path("characters")
    if not os.path.exists(path):
        return []
    return sorted(f[:-5] for f in os.listdir(path) if f.endswith(".json"))


def save_character(name: str, description: str, base_speaker: str) -> tuple:
    """Save a character profile with voice description (notes not editable in UI)."""
    if not name or not name.strip():
        return "Enter a character name first.", gr.update()
    if not description or not description.strip():
        return "Enter a voice description for the character.", gr.update()
    
    safe = _safe_name(name)
    if not safe:
        return "Invalid character name.", gr.update()
    
    path = _preset_path("characters")
    os.makedirs(path, exist_ok=True)

    existing_notes = ""
    existing_path = os.path.join(path, f"{safe}.json")
    if os.path.exists(existing_path):
        try:
            with open(existing_path, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
            existing_notes = (existing.get("notes") or "").strip()
        except Exception:
            existing_notes = ""

    bs = (base_speaker or "").strip()
    if not bs or bs == BASE_SPEAKER_NONE:
        base_speaker_stored = ""
    else:
        base_speaker_stored = _speaker_id_from_ui(bs)
    
    data = {
        "name": name.strip(),
        "description": description.strip(),
        "base_speaker": base_speaker_stored,
        "notes": existing_notes,
        "created": datetime.now().isoformat()
    }
    
    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    characters = get_characters()
    return f"Saved character '{safe}'.", gr.update(choices=characters, value=safe)


def load_character(name: str) -> tuple:
    """Load character profile and return (name, description, base_speaker)."""
    nc = (gr.update(),) * 3
    if not name:
        return nc
    
    filepath = os.path.join(_preset_path("characters"), f"{name}.json")
    if not os.path.exists(filepath):
        return nc
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    bs_raw = (data.get("base_speaker") or "").strip()
    if not bs_raw:
        base_label = BASE_SPEAKER_NONE
    else:
        base_label = _label_for_speaker_id(bs_raw)
    
    return (
        gr.update(value=data.get("name", name)),
        gr.update(value=data.get("description", "")),
        gr.update(value=base_label),
    )


# ── Dialogue Processing ───────────────────────────────────────────────────────

def parse_dialogue(dialogue_text: str) -> list:
    """
    Parse dialogue text and extract character lines.
    Expected format:
    Character1: Line of dialogue
    Character2: Another line
    
    Returns list of tuples: [(character_name, dialogue_line), ...]
    """
    lines = []
    for line in dialogue_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Look for "Character:" pattern
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                character = parts[0].strip()
                dialogue = parts[1].strip()
                if character and dialogue:
                    lines.append((character, dialogue))
    
    return lines


def get_character_profile(character_name: str) -> tuple[str, str | None]:
    """Get (voice_description, base_speaker_id) for a character, with defaults."""
    characters = get_characters()
    if character_name in characters:
        filepath = os.path.join(_preset_path("characters"), f"{character_name}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                desc = data.get("description", f"Neutral voice for {character_name}")
                bs = (data.get("base_speaker") or "").strip()
                return desc, (bs or None)
    
    # Return a basic description if character not found
    return f"Neutral voice for {character_name}", None


def concatenate_audio_files(audio_files: list, output_path: str, pause_duration: float = 0.3) -> bool:
    """
    Concatenate multiple WAV files with optional pauses between them.
    
    Args:
        audio_files: List of paths to WAV files to concatenate
        output_path: Path where the concatenated audio should be saved
        pause_duration: Duration of silence to add between files (in seconds)
    
    Returns:
        True if successful, False otherwise
    """
    if not audio_files:
        return False
    
    try:
        # Read all audio files and collect their data
        audio_data = []
        sample_rate = None
        
        for file_path in audio_files:
            if not os.path.exists(file_path):
                continue
                
            with wave.open(file_path, 'rb') as wav_file:
                if sample_rate is None:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                
                # Read audio data
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data.append(frames)
                
                # Add pause between files (except after the last one)
                if file_path != audio_files[-1] and pause_duration > 0:
                    pause_samples = int(sample_rate * pause_duration * channels)
                    pause_data = b'\x00' * (pause_samples * sample_width)
                    audio_data.append(pause_data)
        
        if not audio_data or sample_rate is None:
            return False
        
        # Write concatenated audio
        with wave.open(output_path, 'wb') as output_wav:
            output_wav.setnchannels(channels)
            output_wav.setsampwidth(sample_width)
            output_wav.setframerate(sample_rate)
            
            for data in audio_data:
                output_wav.writeframes(data)
        
        return True
        
    except Exception as e:
        print(f"Error concatenating audio files: {e}")
        return False


def generate_multi_character_dialogue(dialogue_text: str, lang_code: str, temperature: float, 
                                    autoplay: bool, remove_linebreaks: bool, history: list):
    """Generate audio for multi-character dialogue using Voice Design model."""
    if not dialogue_text or not dialogue_text.strip():
        return None, "Please enter dialogue text.", *_fail(history)
    
    # Parse the dialogue
    dialogue_lines = parse_dialogue(dialogue_text)
    if not dialogue_lines:
        return None, "No valid dialogue found. Use format 'Character: Line of dialogue'", *_fail(history)
    
    model_key = MODE_TO_KEY["design"]  # Use Voice Design model for character voices
    subfolder = "MultiCharacter"
    
    try:
        model = get_model(model_key)
    except RuntimeError as e:
        return None, str(e), *_fail(history)
    
    # Generate audio for each line and combine them
    temp_dir = f"temp_{int(time.time())}"
    combined_audio_files = []
    
    try:
        for i, (character, line) in enumerate(dialogue_lines):
            line_text = _text_for_model(line, remove_linebreaks)
            if not line_text.strip():
                continue
                
            # Get character voice description + optional base speaker
            voice_description, base_speaker_id = get_character_profile(character)
            
            # Create individual temp directory for this line
            line_temp_dir = f"{temp_dir}_line_{i}"
            
            # Generate audio for this line
            kwargs = {
                "model": model,
                "text": line_text,
                "instruct": voice_description,
                "lang_code": lang_code,
                "temperature": temperature,
                "output_path": line_temp_dir,
            }
            if base_speaker_id:
                kwargs["voice"] = base_speaker_id
            generate_audio(**kwargs)
            
            # Check if audio was generated
            line_audio = os.path.join(line_temp_dir, "audio_000.wav")
            if os.path.exists(line_audio):
                combined_audio_files.append(line_audio)
        
        if not combined_audio_files:
            return None, "No audio files were generated.", *_fail(history)
        
        # Save the output
        save_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
        os.makedirs(save_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%H-%M-%S")
        filename = f"{timestamp}_dialogue.wav"
        final_path = os.path.join(save_path, filename)
        
        # Concatenate all audio files
        if len(combined_audio_files) == 1:
            # Single file, just copy it
            shutil.copy(combined_audio_files[0], final_path)
        else:
            # Multiple files, concatenate them with pauses
            success = concatenate_audio_files(combined_audio_files, final_path, pause_duration=0.5)
            if not success:
                return None, "Failed to concatenate audio files.", *_fail(history)
        
        # Clean up temp directories
        for temp in [temp_dir] + [f"{temp_dir}_line_{i}" for i in range(len(dialogue_lines))]:
            if os.path.exists(temp):
                shutil.rmtree(temp, ignore_errors=True)
        
        if os.path.exists(final_path):
            _maybe_play(final_path, autoplay)
            new_history, df_data = _add_to_history(history, final_path, "Multi-Character", dialogue_text[:60])
            
            # Create detailed status message
            characters_used = list(set(char for char, _ in dialogue_lines))
            status_msg = f"Generated dialogue with {len(dialogue_lines)} lines from {len(characters_used)} characters: {', '.join(characters_used)}. Saved to outputs/{subfolder}/{filename}"
            
            return final_path, status_msg, new_history, df_data
        
        return None, "Generation failed: output file not found.", *_fail(history)
        
    except Exception as e:
        # Clean up on error
        for temp in [temp_dir] + [f"{temp_dir}_line_{i}" for i in range(len(dialogue_lines))]:
            if os.path.exists(temp):
                shutil.rmtree(temp, ignore_errors=True)
        return None, f"Error: {e}", *_fail(history)


def get_presets(category: str) -> list:
    path = _preset_path(category)
    if not os.path.exists(path):
        return []
    return sorted(f[:-5] for f in os.listdir(path) if f.endswith(".json"))


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")


# Custom Voice presets — speaker + speed + emotion + text
def save_preset_custom(name: str, speaker: str, speed: str, emotion: str, text: str):
    if not name or not name.strip():
        return "Enter a preset name first.", gr.update()
    safe = _safe_name(name)
    if not safe:
        return "Invalid preset name.", gr.update()
    path = _preset_path("emotion")
    os.makedirs(path, exist_ok=True)
    data = {"speaker": speaker, "speed": speed, "emotion": emotion.strip(), "text": text.strip()}
    with open(os.path.join(path, f"{safe}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    presets = get_presets("emotion")
    return f"Saved preset '{safe}'.", gr.update(choices=presets, value=safe)


def load_preset_custom(name: str):
    """Returns (speaker, speed, emotion, text, preset_name) gr.updates."""
    nc = (gr.update(),) * 5
    if not name:
        return nc
    filepath = os.path.join(_preset_path("emotion"), f"{name}.json")
    if not os.path.exists(filepath):
        return nc
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    sp = data.get("speaker", "vivian")
    speaker_label = sp if sp in SPEAKER_ID_BY_LABEL else _label_for_speaker_id(str(sp))
    return (
        gr.update(value=speaker_label),
        gr.update(value=data.get("speed", "Normal (1.0x)")),
        gr.update(value=data.get("emotion", "")),
        gr.update(value=data.get("text", "")),
        gr.update(value=name),
    )


# Voice Design presets — description + text
def save_preset_design(name: str, description: str, text: str):
    if not name or not name.strip():
        return "Enter a preset name first.", gr.update()
    if not description or not description.strip():
        return "Nothing to save — the description is empty.", gr.update()
    safe = _safe_name(name)
    if not safe:
        return "Invalid preset name.", gr.update()
    path = _preset_path("voice_design")
    os.makedirs(path, exist_ok=True)
    data = {"description": description.strip(), "text": text.strip()}
    with open(os.path.join(path, f"{safe}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    presets = get_presets("voice_design")
    return f"Saved preset '{safe}'.", gr.update(choices=presets, value=safe)


def load_preset_design(name: str):
    """Returns (description, text, preset_name) gr.updates."""
    nc = (gr.update(),) * 3
    if not name:
        return nc
    filepath = os.path.join(_preset_path("voice_design"), f"{name}.json")
    if not os.path.exists(filepath):
        return nc
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        gr.update(value=data.get("description", "")),
        gr.update(value=data.get("text", "")),
        gr.update(value=name),
    )


# ── History helpers ───────────────────────────────────────────────────────────

def _wav_duration_seconds(path: str):
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        if not rate:
            return None
        return frames / float(rate)
    except Exception:
        return None


def _format_duration(seconds) -> str:
    if seconds is None:
        return "—"
    try:
        seconds = float(seconds)
    except Exception:
        return "—"
    if seconds < 0:
        return "—"
    minutes = int(seconds // 60)
    sec = seconds - (minutes * 60)
    return f"{minutes}:{sec:04.1f}"


def _add_to_history(history: list, path: str, mode: str, text: str, preset: str | None = None):
    entry = {
        "path": path,
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": mode,
        "duration": _format_duration(_wav_duration_seconds(path)),
        "preset": (preset or "").strip() or "—",
        "text": text[:60] + ("\u2026" if len(text) > 60 else ""),
    }
    new_history = [entry] + history
    if len(new_history) > 50:
        new_history = new_history[:50]
    return new_history, _history_to_df(new_history)


def _history_to_df(history: list) -> list:
    return [[e["time"], e["mode"], e.get("duration", "—"), e.get("preset", "—"), e["text"]] for e in history]


def _fail(history: list):
    return history, _history_to_df(history)


def select_history_row(history: list, evt: gr.SelectData):
    row_idx = evt.index[0]
    if 0 <= row_idx < len(history):
        return gr.update(value=history[row_idx]["path"])
    return gr.update()


# ── Generation functions ──────────────────────────────────────────────────────

def generate_custom(preset_name, speaker, emotion, speed_label, text, lang_code, temperature, autoplay, remove_linebreaks, separate_by_line, history):
    raw_text = (text or "").strip()
    tts_text = raw_text if separate_by_line else _text_for_model(raw_text, remove_linebreaks)
    if not tts_text.strip():
        yield None, "Please enter some text.", *_fail(history)
        return

    model_key = MODE_TO_KEY["custom"]
    subfolder = MODELS[model_key]["output_subfolder"]
    speed = SPEED_MAP.get(speed_label, 1.0)
    instruct = emotion.strip() or "Normal tone"

    try:
        model = get_model(model_key)
    except RuntimeError as e:
        yield None, str(e), *_fail(history)
        return

    voice_id = _speaker_id_from_ui(speaker)

    if separate_by_line:
        segments = _segments_for_history(tts_text)
        if not segments:
            yield None, "No non-empty lines found.", *_fail(history)
            return

        out_paths: list[str] = []
        last_audio = None
        new_history = history
        df_data = _history_to_df(new_history)

        for i, segment in enumerate(segments):
            temp_dir = f"temp_{int(time.time() * 1000)}_{i}"
            try:
                generate_audio(
                    model=model, text=segment, voice=voice_id,
                    instruct=instruct, speed=speed,
                    lang_code=lang_code, temperature=temperature, output_path=temp_dir,
                )
                out = _save_output(temp_dir, subfolder, segment)
                if out:
                    out_paths.append(out)
                    last_audio = out
                    new_history, df_data = _add_to_history(
                        new_history, out, "Custom Voice", segment, preset=preset_name
                    )
                    _maybe_play(out, autoplay)
                    yield (
                        out,
                        f"Saved {i+1}/{len(segments)} clip(s) to outputs/{subfolder}/ (added to history)",
                        new_history,
                        df_data,
                    )
                else:
                    yield (
                        last_audio,
                        f"Generation failed for line {i+1}: output file not found.",
                        new_history,
                        df_data,
                    )
            except Exception as e:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                yield last_audio, f"Error on line {i+1}: {e}", new_history, df_data
                return

        if out_paths:
            yield (
                out_paths[-1],
                f"Saved {len(out_paths)} clip(s) to outputs/{subfolder}/ (added to history)",
                new_history,
                df_data,
            )
            return

        yield None, "Generation failed: output file not found.", *_fail(history)
        return

    # Normal single generation
    temp_dir = f"temp_{int(time.time())}"
    try:
        generate_audio(
            model=model, text=tts_text, voice=voice_id,
            instruct=instruct, speed=speed,
            lang_code=lang_code, temperature=temperature, output_path=temp_dir,
        )
        out = _save_output(temp_dir, subfolder, tts_text)
        if out:
            _maybe_play(out, autoplay)
            new_history, df_data = _add_to_history(history, out, "Custom Voice", tts_text, preset=preset_name)
            yield out, f"Saved to outputs/{subfolder}/{os.path.basename(out)}", new_history, df_data
            return
        yield None, "Generation failed: output file not found.", *_fail(history)
        return
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        yield None, f"Error: {e}", *_fail(history)
        return


def generate_design(preset_name, voice_description, text, lang_code, temperature, autoplay, remove_linebreaks, separate_by_line, history):
    if not voice_description or not voice_description.strip():
        yield None, "Please describe the voice.", *_fail(history)
        return
    raw_text = (text or "").strip()
    tts_text = raw_text if separate_by_line else _text_for_model(raw_text, remove_linebreaks)
    if not tts_text.strip():
        yield None, "Please enter some text.", *_fail(history)
        return

    model_key = MODE_TO_KEY["design"]
    subfolder = MODELS[model_key]["output_subfolder"]

    try:
        model = get_model(model_key)
    except RuntimeError as e:
        yield None, str(e), *_fail(history)
        return

    if separate_by_line:
        segments = _segments_for_history(tts_text)
        if not segments:
            yield None, "No non-empty lines found.", *_fail(history)
            return

        out_paths: list[str] = []
        last_audio = None
        new_history = history
        df_data = _history_to_df(new_history)

        for i, segment in enumerate(segments):
            temp_dir = f"temp_{int(time.time() * 1000)}_{i}"
            try:
                generate_audio(
                    model=model, text=segment,
                    instruct=voice_description.strip(),
                    lang_code=lang_code, temperature=temperature, output_path=temp_dir,
                )
                out = _save_output(temp_dir, subfolder, segment)
                if out:
                    out_paths.append(out)
                    last_audio = out
                    new_history, df_data = _add_to_history(
                        new_history, out, "Voice Design", segment, preset=preset_name
                    )
                    _maybe_play(out, autoplay)
                    yield (
                        out,
                        f"Saved {i+1}/{len(segments)} clip(s) to outputs/{subfolder}/ (added to history)",
                        new_history,
                        df_data,
                    )
                else:
                    yield (
                        last_audio,
                        f"Generation failed for line {i+1}: output file not found.",
                        new_history,
                        df_data,
                    )
            except Exception as e:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                yield last_audio, f"Error on line {i+1}: {e}", new_history, df_data
                return

        if out_paths:
            yield (
                out_paths[-1],
                f"Saved {len(out_paths)} clip(s) to outputs/{subfolder}/ (added to history)",
                new_history,
                df_data,
            )
            return

        yield None, "Generation failed: output file not found.", *_fail(history)
        return

    temp_dir = f"temp_{int(time.time())}"
    try:
        generate_audio(
            model=model, text=tts_text,
            instruct=voice_description.strip(),
            lang_code=lang_code, temperature=temperature, output_path=temp_dir,
        )
        out = _save_output(temp_dir, subfolder, tts_text)
        if out:
            _maybe_play(out, autoplay)
            new_history, df_data = _add_to_history(history, out, "Voice Design", tts_text, preset=preset_name)
            yield out, f"Saved to outputs/{subfolder}/{os.path.basename(out)}", new_history, df_data
            return
        yield None, "Generation failed: output file not found.", *_fail(history)
        return
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        yield None, f"Error: {e}", *_fail(history)
        return


def generate_clone_saved(voice_name, text, lang_code, temperature, autoplay, remove_linebreaks, separate_by_line, history):
    if not voice_name:
        yield None, "No voice selected. Enroll a voice first.", *_fail(history)
        return
    raw_text = (text or "").strip()
    tts_text = raw_text if separate_by_line else _text_for_model(raw_text, remove_linebreaks)
    if not tts_text.strip():
        yield None, "Please enter some text.", *_fail(history)
        return

    model_key = MODE_TO_KEY["clone"]
    subfolder = MODELS[model_key]["output_subfolder"]

    ref_audio = os.path.join(VOICES_DIR, f"{voice_name}.wav")
    ref_text = "."
    txt_path = os.path.join(VOICES_DIR, f"{voice_name}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip() or "."

    try:
        model = get_model(model_key)
    except RuntimeError as e:
        yield None, str(e), *_fail(history)
        return

    if separate_by_line:
        segments = _segments_for_history(tts_text)
        if not segments:
            yield None, "No non-empty lines found.", *_fail(history)
            return

        out_paths: list[str] = []
        last_audio = None
        new_history = history
        df_data = _history_to_df(new_history)

        for i, segment in enumerate(segments):
            temp_dir = f"temp_{int(time.time() * 1000)}_{i}"
            try:
                generate_audio(
                    model=model, text=segment,
                    ref_audio=ref_audio, ref_text=ref_text,
                    lang_code=lang_code, temperature=temperature, output_path=temp_dir,
                )
                out = _save_output(temp_dir, subfolder, segment)
                if out:
                    out_paths.append(out)
                    last_audio = out
                    new_history, df_data = _add_to_history(new_history, out, f"Clone: {voice_name}", segment)
                    _maybe_play(out, autoplay)
                    yield (
                        out,
                        f"Saved {i+1}/{len(segments)} clip(s) to outputs/{subfolder}/ (added to history)",
                        new_history,
                        df_data,
                    )
                else:
                    yield (
                        last_audio,
                        f"Generation failed for line {i+1}: output file not found.",
                        new_history,
                        df_data,
                    )
            except Exception as e:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                yield last_audio, f"Error on line {i+1}: {e}", new_history, df_data
                return

        if out_paths:
            yield (
                out_paths[-1],
                f"Saved {len(out_paths)} clip(s) to outputs/{subfolder}/ (added to history)",
                new_history,
                df_data,
            )
            return

        yield None, "Generation failed: output file not found.", *_fail(history)
        return

    temp_dir = f"temp_{int(time.time())}"
    try:
        generate_audio(
            model=model, text=tts_text,
            ref_audio=ref_audio, ref_text=ref_text,
            lang_code=lang_code, temperature=temperature, output_path=temp_dir,
        )
        out = _save_output(temp_dir, subfolder, tts_text)
        if out:
            _maybe_play(out, autoplay)
            new_history, df_data = _add_to_history(history, out, f"Clone: {voice_name}", tts_text)
            yield out, f"Saved to outputs/{subfolder}/{os.path.basename(out)}", new_history, df_data
            return
        yield None, "Generation failed: output file not found.", *_fail(history)
        return
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        yield None, f"Error: {e}", *_fail(history)
        return


def generate_clone_quick(ref_audio_path, ref_text, text, lang_code, temperature, autoplay, remove_linebreaks, separate_by_line, history):
    if not ref_audio_path:
        yield None, "Please upload a reference audio file.", *_fail(history)
        return
    raw_text = (text or "").strip()
    tts_text = raw_text if separate_by_line else _text_for_model(raw_text, remove_linebreaks)
    if not tts_text.strip():
        yield None, "Please enter some text.", *_fail(history)
        return

    model_key = MODE_TO_KEY["clone"]
    subfolder = MODELS[model_key]["output_subfolder"]

    converted = convert_audio_if_needed(ref_audio_path)
    if not converted:
        yield None, "Could not read/convert the reference audio. Is ffmpeg installed?", *_fail(history)
        return

    try:
        model = get_model(model_key)
    except RuntimeError as e:
        if converted != ref_audio_path and os.path.exists(converted):
            os.remove(converted)
        yield None, str(e), *_fail(history)
        return

    try:
        if separate_by_line:
            segments = _segments_for_history(tts_text)
            if not segments:
                yield None, "No non-empty lines found.", *_fail(history)
                return

            out_paths: list[str] = []
            last_audio = None
            new_history = history
            df_data = _history_to_df(new_history)

            for i, segment in enumerate(segments):
                temp_dir = f"temp_{int(time.time() * 1000)}_{i}"
                try:
                    generate_audio(
                        model=model, text=segment,
                        ref_audio=converted, ref_text=ref_text.strip() or ".",
                        lang_code=lang_code, temperature=temperature, output_path=temp_dir,
                    )
                    out = _save_output(temp_dir, subfolder, segment)
                    if out:
                        out_paths.append(out)
                        last_audio = out
                        new_history, df_data = _add_to_history(new_history, out, "Quick Clone", segment)
                        _maybe_play(out, autoplay)
                        yield (
                            out,
                            f"Saved {i+1}/{len(segments)} clip(s) to outputs/{subfolder}/ (added to history)",
                            new_history,
                            df_data,
                        )
                    else:
                        yield (
                            last_audio,
                            f"Generation failed for line {i+1}: output file not found.",
                            new_history,
                            df_data,
                        )
                except Exception as e:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    yield last_audio, f"Error on line {i+1}: {e}", new_history, df_data
                    return

            if out_paths:
                yield (
                    out_paths[-1],
                    f"Saved {len(out_paths)} clip(s) to outputs/{subfolder}/ (added to history)",
                    new_history,
                    df_data,
                )
                return

            yield None, "Generation failed: output file not found.", *_fail(history)
            return

        temp_dir = f"temp_{int(time.time())}"
        generate_audio(
            model=model, text=tts_text,
            ref_audio=converted, ref_text=ref_text.strip() or ".",
            lang_code=lang_code, temperature=temperature, output_path=temp_dir,
        )

        out = _save_output(temp_dir, subfolder, tts_text)
        if out:
            _maybe_play(out, autoplay)
            new_history, df_data = _add_to_history(history, out, "Quick Clone", tts_text)
            yield out, f"Saved to outputs/{subfolder}/{os.path.basename(out)}", new_history, df_data
            return
        yield None, "Generation failed: output file not found.", *_fail(history)
        return
    except Exception as e:
        yield None, f"Error: {e}", *_fail(history)
        return
    finally:
        if converted != ref_audio_path and converted and os.path.exists(converted):
            os.remove(converted)


def enroll_voice(voice_name: str, ref_audio_path: str, ref_text: str):
    if not voice_name or not voice_name.strip():
        return "Please enter a voice name.", gr.update()
    if not ref_audio_path:
        return "Please upload a reference audio file.", gr.update()

    safe_name = re.sub(r"[^\w\s-]", "", voice_name).strip().replace(" ", "_")
    if not safe_name:
        return "Invalid voice name.", gr.update()

    converted = convert_audio_if_needed(ref_audio_path)
    if not converted:
        return "Could not read/convert the audio. Is ffmpeg installed?", gr.update()

    os.makedirs(VOICES_DIR, exist_ok=True)
    shutil.copy(converted, os.path.join(VOICES_DIR, f"{safe_name}.wav"))
    with open(os.path.join(VOICES_DIR, f"{safe_name}.txt"), "w", encoding="utf-8") as f:
        f.write(ref_text.strip())

    if converted != ref_audio_path and os.path.exists(converted):
        os.remove(converted)

    voices = get_saved_voices()
    return f"Voice '{safe_name}' enrolled successfully.", gr.update(choices=voices, value=safe_name)


def refresh_voices():
    voices = get_saved_voices()
    return gr.update(choices=voices, value=voices[0] if voices else None)


def save_quick_clone_voice(voice_name: str, ref_audio_path: str, ref_text: str):
    """Save a Quick Clone voice to the library."""
    if not voice_name or not voice_name.strip():
        return "Please enter a voice name to save.", gr.update()
    if not ref_audio_path:
        return "Please upload a reference audio file first.", gr.update()

    # Use the existing enroll_voice function
    result, dropdown_update = enroll_voice(voice_name, ref_audio_path, ref_text)
    return result, dropdown_update


# ── UI ────────────────────────────────────────────────────────────────────────

def _global_settings():
    """Global settings that apply to all generation modes."""
    s = load_app_settings()
    with gr.Row(elem_classes=["global-settings-row"]):
        lang = gr.Dropdown(choices=LANG_CHOICES, value=s["lang"], label="Language", scale=2)
        ap = gr.Checkbox(label="Auto-play when done", value=s["autoplay"], scale=1)
        nlb = gr.Checkbox(
            label="Remove line breaks",
            value=s["remove_linebreaks"],
            scale=2,
        )
        sbl = gr.Checkbox(
            label="Separate by line",
            value=s.get("separate_by_line", False),
            scale=2,
        )
    with gr.Accordion("Advanced Settings", open=False, visible=False):
        temp = gr.Slider(
            minimum=0.1, maximum=1.5, value=s["temperature"], step=0.05,
            label="Sampling temperature (default 0.7)",
        )
    return lang, temp, ap, nlb, sbl


def build_ui():
    saved_voices = get_saved_voices()
    emotion_placeholder = "e.g. " + " / ".join(EMOTION_EXAMPLES[:2])
    emotion_presets = get_presets("emotion")
    design_presets = get_presets("voice_design")
    characters = get_characters()

    with gr.Blocks(theme=gr.themes.Default(), title="Qwen3-TTS", css=CSS) as demo:

        history_state = gr.State([])
        settings_state = gr.State(load_app_settings())

        gr.Markdown("# Qwen3-TTS\nLocal AI text-to-speech on Apple Silicon via MLX.")
        
        # Global settings that apply to all tabs
        global_lang, global_temp, global_ap, global_nlb, global_sbl = _global_settings()

        with gr.Tabs():

            # ── Custom Voice ───────────────────────────────────────────────
            with gr.Tab("Custom Voice"):
                with gr.Row():
                    cv_preset_dd = gr.Dropdown(
                        choices=emotion_presets, value=None, label="Load preset",
                        scale=3, interactive=True,
                    )
                    cv_preset_name = gr.Textbox(
                        label="Save as", placeholder="Preset name", scale=2, lines=1,
                    )
                    cv_preset_save = gr.Button("Save", scale=1)
                with gr.Row():
                    cv_speaker = gr.Dropdown(
                        choices=SPEAKER_LABELS, value=SPEAKER_LABELS[0], label="Speaker",
                    )
                    cv_speed = gr.Radio(
                        choices=list(SPEED_MAP.keys()), value="Normal (1.0x)", label="Speed",
                    )
                cv_emotion = gr.Textbox(
                    label="Emotion / Style instruction",
                    placeholder=emotion_placeholder,
                    lines=2,
                )
                cv_text = gr.Textbox(
                    label="Text to speak", placeholder="Enter your text here...", lines=4,
                )
                cv_btn = gr.Button("Generate", variant="primary")
                cv_audio = gr.Audio(label="Output", type="filepath", interactive=False)
                cv_status = gr.Textbox(label="Status", interactive=False, lines=1)

            # ── Voice Design ───────────────────────────────────────────────
            with gr.Tab("Voice Design"):
                with gr.Row():
                    vd_preset_dd = gr.Dropdown(
                        choices=design_presets, value=None, label="Load preset",
                        scale=3, interactive=True,
                    )
                    vd_preset_name = gr.Textbox(
                        label="Save as", placeholder="Preset name", scale=2, lines=1,
                    )
                    vd_preset_save = gr.Button("Save", scale=1)
                vd_description = gr.Textbox(
                    label="Voice description",
                    placeholder="e.g. calm British male narrator with a deep voice",
                    lines=2,
                )
                vd_text = gr.Textbox(
                    label="Text to speak", placeholder="Enter your text here...", lines=4,
                )
                vd_btn = gr.Button("Generate", variant="primary")
                vd_audio = gr.Audio(label="Output", type="filepath", interactive=False)
                vd_status = gr.Textbox(label="Status", interactive=False, lines=1)

            # ── Voice Cloning ──────────────────────────────────────────────
            with gr.Tab("Voice Cloning"):
                with gr.Tabs():

                    with gr.Tab("Saved Voices"):
                        with gr.Row():
                            sv_dropdown = gr.Dropdown(
                                choices=saved_voices,
                                value=saved_voices[0] if saved_voices else None,
                                label="Saved Voice", scale=3,
                            )
                            sv_refresh = gr.Button("Refresh", scale=1)
                        sv_text = gr.Textbox(
                            label="Text to speak", placeholder="Enter your text here...", lines=4,
                        )
                        sv_btn = gr.Button("Generate", variant="primary")
                        sv_audio = gr.Audio(label="Output", type="filepath", interactive=False)
                        sv_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        with gr.Accordion("Enroll New Voice", open=False):
                            gr.Markdown(
                                "Save a voice to the library so it appears in the dropdown above. "
                                "A clean 5-10 s clip with a matching transcript gives the best results."
                            )
                            en_name = gr.Textbox(
                                label="Voice name", placeholder="e.g. Boss, Mom, Narrator", lines=1,
                            )
                            en_audio = gr.Audio(
                                label="Reference audio", type="filepath", sources=["upload"],
                            )
                            en_transcript = gr.Textbox(
                                label="Transcript — type EXACTLY what the audio says",
                                placeholder="The quick brown fox jumps over the lazy dog.",
                                lines=2,
                            )
                            en_btn = gr.Button("Enroll Voice", variant="primary")
                            en_status = gr.Textbox(label="Status", interactive=False, lines=1)

                    with gr.Tab("Quick Clone"):
                        gr.Markdown(
                            "Upload any audio clip (5-10 s works best) and optionally provide "
                            "the exact transcript of what is said. You can also save the voice to your library."
                        )
                        qc_audio_in = gr.Audio(
                            label="Reference audio", type="filepath", sources=["upload"],
                        )
                        qc_ref_text = gr.Textbox(
                            label="Transcript (optional but improves quality)",
                            placeholder="Type exactly what the reference audio says...",
                            lines=2,
                        )
                        
                        with gr.Accordion("Save Voice to Library", open=False):
                            qc_save_name = gr.Textbox(
                                label="Voice name (optional)", 
                                placeholder="e.g. Boss, Mom, Narrator", 
                                lines=1
                            )
                            qc_save_btn = gr.Button("Save Voice to Library")
                            qc_save_status = gr.Textbox(label="Save Status", interactive=False, lines=1)
                        
                        qc_text = gr.Textbox(
                            label="Text to speak", placeholder="Enter your text here...", lines=4,
                        )
                        qc_btn = gr.Button("Generate", variant="primary")
                        qc_audio_out = gr.Audio(label="Output", type="filepath", interactive=False)
                        qc_status = gr.Textbox(label="Status", interactive=False, lines=1)

            # ── Multi-Character Dialogue ───────────────────────────────────────
            with gr.Tab("Multi-Character Dialogue"):
                gr.Markdown(
                    "Create conversations between multiple characters. Each character can have "
                    "their own unique voice profile. Use format: `Character: Line of dialogue`"
                )
                
                with gr.Accordion("Character Management", open=False):
                    with gr.Row():
                        char_dropdown = gr.Dropdown(
                            choices=characters, value=None, label="Load Character",
                            scale=2, interactive=True,
                        )
                        char_refresh = gr.Button("Refresh", scale=1)
                    
                    char_name = gr.Textbox(
                        label="Character Name", placeholder="e.g. Lucas, Mia", lines=1
                    )
                    char_description = gr.Textbox(
                        label="Voice Description",
                        placeholder="e.g. Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous",
                        lines=3
                    )
                    char_base_speaker = gr.Dropdown(
                        choices=BASE_SPEAKER_DROPDOWN_CHOICES,
                        value=BASE_SPEAKER_NONE,
                        label="Base Speaker (optional)",
                    )
                    
                    with gr.Row():
                        char_save = gr.Button("Save Character", variant="primary")
                        char_status = gr.Textbox(label="Status", interactive=False, lines=1, scale=2)
                
                dialogue_text = gr.Textbox(
                    label="Dialogue Script",
                    placeholder="""Lucas: H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?
Mia: Oh wow, my mortal enemy - Mr. Thompson's problem sets. Thanks for rescuing me from that F.
Lucas: No problem! I actually... kinda finished those already? If you want to compare answers or something...
Mia: Is this your sneaky way of saying you want to study together, Lucas?""",
                    lines=8
                )
                
                dialogue_btn = gr.Button("Generate Dialogue", variant="primary")
                dialogue_audio = gr.Audio(label="Output", type="filepath", interactive=False)
                dialogue_status = gr.Textbox(label="Status", interactive=False, lines=2)

        # ── History ────────────────────────────────────────────────────────
        with gr.Accordion("Generation History", open=True):
            history_df = gr.Dataframe(
                headers=["Time", "Mode", "Clip length", "Preset", "Text preview"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False, wrap=True, label=None,
            )
            history_player = gr.Audio(
                label="Click a row above to play", type="filepath", interactive=False,
            )

        # ── Event handlers ─────────────────────────────────────────────────

        def _persist_global_settings(lang, temp, ap, nlb, sbl, _current):
            s = {
                "lang": lang,
                "temperature": _clamp(temp, 0.1, 1.5),
                "autoplay": bool(ap),
                "remove_linebreaks": bool(nlb),
                "separate_by_line": bool(sbl),
            }
            if s["remove_linebreaks"] and s["separate_by_line"]:
                s["separate_by_line"] = False
            save_app_settings(s)
            return s

        global_lang.change(
            fn=_persist_global_settings,
            inputs=[global_lang, global_temp, global_ap, global_nlb, global_sbl, settings_state],
            outputs=[settings_state],
        )
        global_temp.change(
            fn=_persist_global_settings,
            inputs=[global_lang, global_temp, global_ap, global_nlb, global_sbl, settings_state],
            outputs=[settings_state],
        )
        global_ap.change(
            fn=_persist_global_settings,
            inputs=[global_lang, global_temp, global_ap, global_nlb, global_sbl, settings_state],
            outputs=[settings_state],
        )

        def _persist_from_nlb(lang, temp, ap, nlb, sbl, _current):
            # If user turns on "Remove line breaks", "Separate by line" must turn off.
            if bool(nlb) and bool(sbl):
                sbl = False
            s = _persist_global_settings(lang, temp, ap, nlb, sbl, _current)
            return s, gr.update(value=bool(sbl))

        def _persist_from_sbl(lang, temp, ap, nlb, sbl, _current):
            # If user turns on "Separate by line", force "Remove line breaks" off.
            if bool(sbl) and bool(nlb):
                nlb = False
            s = _persist_global_settings(lang, temp, ap, nlb, sbl, _current)
            return s, gr.update(value=bool(nlb))

        global_nlb.change(
            fn=_persist_from_nlb,
            inputs=[global_lang, global_temp, global_ap, global_nlb, global_sbl, settings_state],
            outputs=[settings_state, global_sbl],
        )
        global_sbl.change(
            fn=_persist_from_sbl,
            inputs=[global_lang, global_temp, global_ap, global_nlb, global_sbl, settings_state],
            outputs=[settings_state, global_nlb],
        )

        # Preset: load
        cv_preset_dd.change(
            fn=load_preset_custom,
            inputs=[cv_preset_dd],
            outputs=[cv_speaker, cv_speed, cv_emotion, cv_text, cv_preset_name],
        )
        vd_preset_dd.change(
            fn=load_preset_design,
            inputs=[vd_preset_dd],
            outputs=[vd_description, vd_text, vd_preset_name],
        )

        # Preset: save
        cv_preset_save.click(
            fn=save_preset_custom,
            inputs=[cv_preset_name, cv_speaker, cv_speed, cv_emotion, cv_text],
            outputs=[cv_status, cv_preset_dd],
        )
        vd_preset_save.click(
            fn=save_preset_design,
            inputs=[vd_preset_name, vd_description, vd_text],
            outputs=[vd_status, vd_preset_dd],
        )

        # Generate
        cv_btn.click(
            fn=generate_custom,
            inputs=[cv_preset_dd, cv_speaker, cv_emotion, cv_speed, cv_text, global_lang, global_temp, global_ap, global_nlb, global_sbl, history_state],
            outputs=[cv_audio, cv_status, history_state, history_df],
        )
        vd_btn.click(
            fn=generate_design,
            inputs=[vd_preset_dd, vd_description, vd_text, global_lang, global_temp, global_ap, global_nlb, global_sbl, history_state],
            outputs=[vd_audio, vd_status, history_state, history_df],
        )
        sv_btn.click(
            fn=generate_clone_saved,
            inputs=[sv_dropdown, sv_text, global_lang, global_temp, global_ap, global_nlb, global_sbl, history_state],
            outputs=[sv_audio, sv_status, history_state, history_df],
        )
        qc_btn.click(
            fn=generate_clone_quick,
            inputs=[qc_audio_in, qc_ref_text, qc_text, global_lang, global_temp, global_ap, global_nlb, global_sbl, history_state],
            outputs=[qc_audio_out, qc_status, history_state, history_df],
        )
        qc_save_btn.click(
            fn=save_quick_clone_voice,
            inputs=[qc_save_name, qc_audio_in, qc_ref_text],
            outputs=[qc_save_status, sv_dropdown],
        )

        en_btn.click(
            fn=enroll_voice,
            inputs=[en_name, en_audio, en_transcript],
            outputs=[en_status, sv_dropdown],
        )
        sv_refresh.click(fn=refresh_voices, outputs=[sv_dropdown])

        # Multi-Character Dialogue handlers
        char_dropdown.change(
            fn=load_character,
            inputs=[char_dropdown],
            outputs=[char_name, char_description, char_base_speaker],
        )
        char_save.click(
            fn=save_character,
            inputs=[char_name, char_description, char_base_speaker],
            outputs=[char_status, char_dropdown],
        )
        char_refresh.click(
            fn=lambda: gr.update(choices=get_characters()),
            outputs=[char_dropdown]
        )
        dialogue_btn.click(
            fn=generate_multi_character_dialogue,
            inputs=[dialogue_text, global_lang, global_temp, global_ap, global_nlb, history_state],
            outputs=[dialogue_audio, dialogue_status, history_state, history_df],
        )

        history_df.select(
            fn=select_history_row,
            inputs=[history_state],
            outputs=[history_player],
        )

    return demo


if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    demo = build_ui()
    demo.launch()
