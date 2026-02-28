import os
import librosa
import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
from TTS.api import TTS

# =========================
# CONFIG
# =========================

RAW_DIR = "voxforge_raw"
OUT_REAL = "data/real"
OUT_FAKE = "data/fake"

TARGET_SR = 16000
MIN_DUR = 2.5
MAX_DUR = 8.0
MAX_FILES = 5000

os.makedirs(OUT_REAL, exist_ok=True)

engines = ["silero", "coqui", "ruslan", "mailabs"]
for e in engines:
    os.makedirs(os.path.join(OUT_FAKE, e), exist_ok=True)

# =========================
# LOAD TTS
# =========================

print("Loading TTS models...")

silero_model = None
silero_utils = None
coqui = None

try:
    silero_model, silero_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v3_1_ru'
    )
    print("Silero TTS loaded")
except Exception as e:
    print("Warning: failed to load Silero TTS:", e)

try:
    coqui = TTS(model_name="tts_models/ru/v3_1/ru_v3_1", progress_bar=False)
    print("Coqui TTS loaded")
except Exception as e:
    print("Warning: failed to load Coqui TTS:", e)

# RUSLAN & M-AILABS можно подключить аналогично через Coqui
# если у тебя есть их checkpoint — просто добавь model_name

# =========================
# AUDIO UTILS
# =========================

def process_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)

    if sr != TARGET_SR:
        y = librosa.resample(y, sr, TARGET_SR)

    dur = len(y) / TARGET_SR
    if not (MIN_DUR <= dur <= MAX_DUR):
        return None

    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y / peak * 0.98

    return y

def save_wav(path, audio):
    sf.write(path, audio, TARGET_SR, subtype="PCM_16")

# =========================
# TTS GENERATION
# =========================

def generate_tts(engine, text):
    try:
        if engine == "silero":
            if silero_model is None:
                return None
            audio = silero_model.apply_tts(
                text=text,
                speaker='baya',
                sample_rate=48000
            )
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = librosa.resample(audio, 48000, TARGET_SR)

        elif engine == "coqui":
            if coqui is None:
                return None
            audio = coqui.tts(text)
            audio = np.array(audio)
            # try to get sample rate from model utils if available
            src_sr = 22050
            try:
                src_sr = getattr(coqui, "sample_rate", src_sr)
            except Exception:
                pass
            if src_sr != TARGET_SR:
                audio = librosa.resample(audio, src_sr, TARGET_SR)

        else:
            # not implemented engines (ruslan, mailabs)
            return None
    except Exception as e:
        print(f"TTS generation failed for {engine}:", e)
        return None

    if len(audio) / TARGET_SR > MAX_DUR:
        audio = audio[:int(MAX_DUR * TARGET_SR)]

    return audio

# =========================
# MAIN PIPELINE
# =========================

utt_counter = 0

for speaker_folder in tqdm(os.listdir(RAW_DIR)):

    if utt_counter >= MAX_FILES:
        break

    speaker_path = os.path.join(RAW_DIR, speaker_folder)
    etc_path = os.path.join(speaker_path, "etc")
    wav_path = os.path.join(speaker_path, "wav")

    if not os.path.exists(etc_path):
        continue

    prompts_file = os.path.join(etc_path, "PROMPTS")
    if not os.path.exists(prompts_file):
        continue

    # load prompts
    prompts = {}
    with open(prompts_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                prompts[parts[0]] = parts[1]

    if not os.path.isdir(wav_path):
        continue

    for wav_file in os.listdir(wav_path):

        if utt_counter >= MAX_FILES:
            break

        if not wav_file.lower().endswith('.wav'):
            continue

        key = os.path.splitext(wav_file)[0]
        if key not in prompts:
            continue

        text = prompts[key]
        src_wav = os.path.join(wav_path, wav_file)

        audio = process_audio(src_wav)
        if audio is None:
            continue

        utt_id = f"utt_{utt_counter:06d}"
        real_out = os.path.join(OUT_REAL, f"{utt_id}.wav")
        save_wav(real_out, audio)

        # generate fake
        for engine in engines:
            fake_audio = generate_tts(engine, text)
            if fake_audio is None:
                continue

            fake_out = os.path.join(OUT_FAKE, engine, f"{utt_id}.wav")
            save_wav(fake_out, fake_audio)

        utt_counter += 1

print("DONE.")
print("Total utterances:", utt_counter)