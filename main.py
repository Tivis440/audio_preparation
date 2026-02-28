import os
import re
import tarfile
import requests
import tempfile
import shutil
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from TTS.api import TTS
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# =========================
# CONFIG
# =========================

BASE_URL = "http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/"
OUT_REAL = "data/real"
OUT_FAKE = "data/fake"

TARGET_SR = 16000
MIN_DUR = 2.5
MAX_DUR = 8.0
MAX_FILES = 5000
MIN_SPEAKERS = 100

os.makedirs(OUT_REAL, exist_ok=True)

engines = ["silero", "coqui", "ruslan", "mailabs"]
for e in engines:
    os.makedirs(os.path.join(OUT_FAKE, e), exist_ok=True)

# =========================
# DOWNLOAD UTILS
# =========================

def get_archive_links():
    """Fetch all .tgz archive links from VoxForge"""
    try:
        r = requests.get(BASE_URL, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a"):
            href = a.get("href")
            if href and href.endswith(".tgz"):
                links.append(urljoin(BASE_URL, href))
        return links
    except Exception as e:
        print("Warning: failed to fetch archive links:", e)
        return []

def download_file(url, dst):
    """Download file from URL"""
    try:
        r = requests.get(url, stream=True, timeout=30)
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Warning: failed to download {url}:", e)
        raise

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

def has_clipping(audio):
    return np.max(np.abs(audio)) >= 0.999

def rms(audio):
    return np.sqrt(np.mean(audio**2))

def process_audio(path):
    try:
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

        duration = len(y) / TARGET_SR
        if duration < MIN_DUR or duration > MAX_DUR:
            return None

        if has_clipping(y):
            return None

        if rms(y) < 0.01:
            return None

        return y
    except Exception as e:
        print(f"Warning: failed to process {path}:", e)
        return None

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

def main():
    archive_links = get_archive_links()
    if not archive_links:
        print("Error: no archive links found")
        return

    np.random.shuffle(archive_links)

    utt_counter = 0
    speakers = set()
    metadata = []

    for link in tqdm(archive_links, desc="Processing archives"):
        if utt_counter >= MAX_FILES and len(speakers) >= MIN_SPEAKERS:
            break

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = os.path.join(tmpdir, "data.tgz")
                download_file(link, archive_path)

                with tarfile.open(archive_path) as tar:
                    tar.extractall(tmpdir)

                extracted = os.listdir(tmpdir)
                extracted = [d for d in extracted if os.path.isdir(os.path.join(tmpdir, d)) and d != "__MACOSX"]
                if not extracted:
                    continue

                speaker_id = extracted[0]
                speakers.add(speaker_id)

                wav_dir = os.path.join(tmpdir, speaker_id, "wav")
                etc_dir = os.path.join(tmpdir, speaker_id, "etc")

                # Load prompts
                prompts = {}
                prompts_path = os.path.join(etc_dir, "prompts.txt")
                if os.path.exists(prompts_path):
                    with open(prompts_path, encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2:
                                prompts[parts[0]] = parts[1]

                if not os.path.isdir(wav_dir):
                    continue

                for wav_file in os.listdir(wav_dir):
                    if utt_counter >= MAX_FILES:
                        break

                    if not wav_file.lower().endswith(".wav"):
                        continue

                    wav_path = os.path.join(wav_dir, wav_file)
                    audio = process_audio(wav_path)
                    if audio is None:
                        continue

                    utt_id = f"utt_{utt_counter:06d}"
                    real_out = os.path.join(OUT_REAL, f"{utt_id}.wav")
                    sf.write(real_out, audio, TARGET_SR, subtype="PCM_16")

                    text_key = os.path.splitext(wav_file)[0]
                    text = prompts.get(text_key, "")

                    # Generate fake audio
                    for engine in engines:
                        fake_audio = generate_tts(engine, text)
                        if fake_audio is None:
                            continue

                        fake_out = os.path.join(OUT_FAKE, engine, f"{utt_id}.wav")
                        sf.write(fake_out, fake_audio, TARGET_SR, subtype="PCM_16")

                    metadata.append({
                        "utt_id": utt_id,
                        "speaker": speaker_id,
                        "text": text
                    })

                    utt_counter += 1

        except Exception as e:
            print(f"Warning: error processing archive {link}:", e)
            continue

    # Save metadata
    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv("metadata.csv", index=False)

    print("Done.")
    print(f"Files: {utt_counter}")
    print(f"Speakers: {len(speakers)}")

if __name__ == "__main__":
    main()