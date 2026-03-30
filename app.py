import os
import sys
import shutil
import subprocess
import threading
import time
import re
import json
import warnings

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

MODE_TO_KEY = {"custom": "1", "design": "2", "clone": "3"}

SPEED_MAP = {"Normal (1.0x)": 1.0, "Fast (1.3x)": 1.3, "Slow (0.8x)": 0.8}

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
]

SPEAKER_LABELS = [label for label, _ in SPEAKER_CHOICES]
SPEAKER_ID_BY_LABEL = {label: sid for label, sid in SPEAKER_CHOICES}


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


CSS = ""


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
    filename = f"{timestamp}_{clean_text}.wav"
    final_path = os.path.join(save_path, filename)

    source = os.path.join(temp_dir, "audio_000.wav")
    if os.path.exists(source):
        shutil.move(source, final_path)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return final_path if os.path.exists(final_path) else None


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


def save_character(name: str, description: str, base_speaker: str = "vivian", notes: str = "") -> tuple:
    """Save a character profile with voice description."""
    if not name or not name.strip():
        return "Enter a character name first.", gr.update()
    if not description or not description.strip():
        return "Enter a voice description for the character.", gr.update()
    
    safe = _safe_name(name)
    if not safe:
        return "Invalid character name.", gr.update()
    
    path = _preset_path("characters")
    os.makedirs(path, exist_ok=True)
    
    data = {
        "name": name.strip(),
        "description": description.strip(),
        "base_speaker": base_speaker,
        "notes": notes.strip(),
        "created": datetime.now().isoformat()
    }
    
    with open(os.path.join(path, f"{safe}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    characters = get_characters()
    return f"Saved character '{safe}'.", gr.update(choices=characters, value=safe)


def load_character(name: str) -> tuple:
    """Load character profile and return (name, description, base_speaker, notes)."""
    nc = (gr.update(),) * 4
    if not name:
        return nc
    
    filepath = os.path.join(_preset_path("characters"), f"{name}.json")
    if not os.path.exists(filepath):
        return nc
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return (
        gr.update(value=data.get("name", name)),
        gr.update(value=data.get("description", "")),
        gr.update(value=data.get("base_speaker", "vivian")),
        gr.update(value=data.get("notes", ""))
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


def get_character_voice_description(character_name: str) -> str:
    """Get the voice description for a character, or return a default."""
    characters = get_characters()
    if character_name in characters:
        filepath = os.path.join(_preset_path("characters"), f"{character_name}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("description", f"Neutral voice for {character_name}")
    
    # Return a basic description if character not found
    return f"Neutral voice for {character_name}"


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

def _add_to_history(history: list, path: str, mode: str, text: str):
    entry = {
        "path": path,
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": mode,
        "text": text[:60] + ("\u2026" if len(text) > 60 else ""),
    }
    new_history = [entry] + history
    if len(new_history) > 50:
        new_history = new_history[:50]
    return new_history, _history_to_df(new_history)


def _history_to_df(history: list) -> list:
    return [[e["time"], e["mode"], e["text"]] for e in history]


def _fail(history: list):
    return history, _history_to_df(history)


def select_history_row(history: list, evt: gr.SelectData):
    row_idx = evt.index[0]
    if 0 <= row_idx < len(history):
        return gr.update(value=history[row_idx]["path"])
    return gr.update()


# ── Generation functions ──────────────────────────────────────────────────────

def generate_custom(speaker, emotion, speed_label, text, lang_code, temperature, autoplay, remove_linebreaks, history):
    tts_text = _text_for_model(text or "", remove_linebreaks)
    if not tts_text.strip():
        return None, "Please enter some text.", *_fail(history)

    model_key = MODE_TO_KEY["custom"]
    subfolder = MODELS[model_key]["output_subfolder"]
    speed = SPEED_MAP.get(speed_label, 1.0)
    instruct = emotion.strip() or "Normal tone"

    try:
        model = get_model(model_key)
    except RuntimeError as e:
        return None, str(e), *_fail(history)

    voice_id = _speaker_id_from_ui(speaker)
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
            new_history, df_data = _add_to_history(history, out, "Custom Voice", tts_text)
            return out, f"Saved to outputs/{subfolder}/{os.path.basename(out)}", new_history, df_data
        return None, "Generation failed: output file not found.", *_fail(history)
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, f"Error: {e}", *_fail(history)


def generate_design(voice_description, text, lang_code, temperature, autoplay, remove_linebreaks, history):
    if not voice_description or not voice_description.strip():
        return None, "Please describe the voice.", *_fail(history)
    tts_text = _text_for_model(text or "", remove_linebreaks)
    if not tts_text.strip():
        return None, "Please enter some text.", *_fail(history)

    model_key = MODE_TO_KEY["design"]
    subfolder = MODELS[model_key]["output_subfolder"]

    try:
        model = get_model(model_key)
    except RuntimeError as e:
        return None, str(e), *_fail(history)

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
            new_history, df_data = _add_to_history(history, out, "Voice Design", tts_text)
            return out, f"Saved to outputs/{subfolder}/{os.path.basename(out)}", new_history, df_data
        return None, "Generation failed: output file not found.", *_fail(history)
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, f"Error: {e}", *_fail(history)


def generate_clone_saved(voice_name, text, lang_code, temperature, autoplay, remove_linebreaks, history):
    if not voice_name:
        return None, "No voice selected. Enroll a voice first.", *_fail(history)
    tts_text = _text_for_model(text or "", remove_linebreaks)
    if not tts_text.strip():
        return None, "Please enter some text.", *_fail(history)

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
        return None, str(e), *_fail(history)

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
            return out, f"Saved to outputs/{subfolder}/{os.path.basename(out)}", new_history, df_data
        return None, "Generation failed: output file not found.", *_fail(history)
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, f"Error: {e}", *_fail(history)


def generate_clone_quick(ref_audio_path, ref_text, text, lang_code, temperature, autoplay, remove_linebreaks, history):
    if not ref_audio_path:
        return None, "Please upload a reference audio file.", *_fail(history)
    tts_text = _text_for_model(text or "", remove_linebreaks)
    if not tts_text.strip():
        return None, "Please enter some text.", *_fail(history)

    model_key = MODE_TO_KEY["clone"]
    subfolder = MODELS[model_key]["output_subfolder"]

    converted = convert_audio_if_needed(ref_audio_path)
    if not converted:
        return None, "Could not read/convert the reference audio. Is ffmpeg installed?", *_fail(history)

    try:
        model = get_model(model_key)
    except RuntimeError as e:
        return None, str(e), *_fail(history)

    temp_dir = f"temp_{int(time.time())}"
    try:
        generate_audio(
            model=model, text=tts_text,
            ref_audio=converted, ref_text=ref_text.strip() or ".",
            lang_code=lang_code, temperature=temperature, output_path=temp_dir,
        )
        out = _save_output(temp_dir, subfolder, tts_text)
        if converted != ref_audio_path and os.path.exists(converted):
            os.remove(converted)
        if out:
            _maybe_play(out, autoplay)
            new_history, df_data = _add_to_history(history, out, "Quick Clone", tts_text)
            return out, f"Saved to outputs/{subfolder}/{os.path.basename(out)}", new_history, df_data
        return None, "Generation failed: output file not found.", *_fail(history)
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, f"Error: {e}", *_fail(history)


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


# ── UI ────────────────────────────────────────────────────────────────────────

def _settings_row():
    """Language + options row; temperature in a collapsible accordion."""
    with gr.Row():
        lang = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language", scale=2)
        ap = gr.Checkbox(label="Auto-play when done", value=False, scale=1)
        nlb = gr.Checkbox(
            label="Remove line breaks",
            value=False,
            scale=2,
        )
    with gr.Accordion("Temperature", open=False):
        temp = gr.Slider(
            minimum=0.1, maximum=1.5, value=0.7, step=0.05,
            label="Sampling temperature (default 0.7)",
        )
    return lang, temp, ap, nlb


def build_ui():
    saved_voices = get_saved_voices()
    emotion_placeholder = "e.g. " + " / ".join(EMOTION_EXAMPLES[:2])
    emotion_presets = get_presets("emotion")
    design_presets = get_presets("voice_design")

    with gr.Blocks(theme=gr.themes.Default(), title="Qwen3-TTS", css=CSS) as demo:

        history_state = gr.State([])

        gr.Markdown("# Qwen3-TTS\nLocal AI text-to-speech on Apple Silicon via MLX.")

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
                cv_lang, cv_temp, cv_ap, cv_nlb = _settings_row()
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
                vd_lang, vd_temp, vd_ap, vd_nlb = _settings_row()
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
                        sv_lang, sv_temp, sv_ap, sv_nlb = _settings_row()
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
                            "the exact transcript of what is said. Not saved to the library."
                        )
                        qc_audio_in = gr.Audio(
                            label="Reference audio", type="filepath", sources=["upload"],
                        )
                        qc_ref_text = gr.Textbox(
                            label="Transcript (optional but improves quality)",
                            placeholder="Type exactly what the reference audio says...",
                            lines=2,
                        )
                        qc_text = gr.Textbox(
                            label="Text to speak", placeholder="Enter your text here...", lines=4,
                        )
                        qc_lang, qc_temp, qc_ap, qc_nlb = _settings_row()
                        qc_btn = gr.Button("Generate", variant="primary")
                        qc_audio_out = gr.Audio(label="Output", type="filepath", interactive=False)
                        qc_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── History ────────────────────────────────────────────────────────
        with gr.Accordion("Generation History", open=True):
            history_df = gr.Dataframe(
                headers=["Time", "Mode", "Text preview"],
                datatype=["str", "str", "str"],
                interactive=False, wrap=True, label=None,
            )
            history_player = gr.Audio(
                label="Click a row above to play", type="filepath", interactive=False,
            )

        # ── Event handlers ─────────────────────────────────────────────────

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
            inputs=[cv_speaker, cv_emotion, cv_speed, cv_text, cv_lang, cv_temp, cv_ap, cv_nlb, history_state],
            outputs=[cv_audio, cv_status, history_state, history_df],
        )
        vd_btn.click(
            fn=generate_design,
            inputs=[vd_description, vd_text, vd_lang, vd_temp, vd_ap, vd_nlb, history_state],
            outputs=[vd_audio, vd_status, history_state, history_df],
        )
        sv_btn.click(
            fn=generate_clone_saved,
            inputs=[sv_dropdown, sv_text, sv_lang, sv_temp, sv_ap, sv_nlb, history_state],
            outputs=[sv_audio, sv_status, history_state, history_df],
        )
        qc_btn.click(
            fn=generate_clone_quick,
            inputs=[qc_audio_in, qc_ref_text, qc_text, qc_lang, qc_temp, qc_ap, qc_nlb, history_state],
            outputs=[qc_audio_out, qc_status, history_state, history_df],
        )

        en_btn.click(
            fn=enroll_voice,
            inputs=[en_name, en_audio, en_transcript],
            outputs=[en_status, sv_dropdown],
        )
        sv_refresh.click(fn=refresh_voices, outputs=[sv_dropdown])

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
