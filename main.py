import os
import tarfile
import requests
import tempfile
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
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

COQUI_MODEL_CANDIDATES = [
    "tts_models/ru/v3_1/ru_v3_1",
    "tts_models/multilingual/multi-dataset/xtts_v2",
]
COQUI_LANGUAGE = "ru"

# =========================
# DOWNLOAD UTILS
# =========================

def safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    base_path = os.path.abspath(path)
    for member in tar.getmembers():
        target_path = os.path.abspath(os.path.join(base_path, member.name))
        if os.path.commonpath([base_path, target_path]) != base_path:
            raise ValueError(f"Unsafe path in archive: {member.name}")
    tar.extractall(path)

def get_archive_links():
    """Fetch all .tgz archive links from VoxForge"""
    try:
        r = requests.get(BASE_URL, timeout=10)
        r.raise_for_status()
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
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Warning: failed to download {url}:", e)
        raise

silero_model = None
silero_utils = None
coqui = None
coqui_language = None
coqui_speaker = None
_tts_loaded = False

def ensure_tts_loaded():
    global silero_model, silero_utils, coqui, coqui_language, coqui_speaker, _tts_loaded
    if _tts_loaded:
        return
    _tts_loaded = True

    print("Loading TTS models...")

    try:
        silero_model, silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v3_1_ru'
        )
        print("✅ Silero TTS loaded")
    except Exception as e:
        print("⚠️  Failed to load Silero TTS:", str(e)[:100])

    try:
        from TTS.api import TTS
        for model_name in COQUI_MODEL_CANDIDATES:
            try:
                coqui = TTS(model_name=model_name, progress_bar=False, gpu=False)
                if "multilingual" in model_name:
                    coqui_language = COQUI_LANGUAGE
                    speakers = getattr(coqui, "speakers", None) or []
                    if speakers:
                        coqui_speaker = speakers[0]
                print(f"✅ Coqui TTS loaded ({model_name})")
                break
            except Exception as e:
                print(f"⚠️  Failed to initialize Coqui model {model_name}:", str(e)[:100])
                coqui = None
    except ModuleNotFoundError as e:
        if e.name == "torchaudio":
            print("⚠️  Failed to load Coqui TTS: missing dependency 'torchaudio'. Install requirements again.")
        else:
            print("⚠️  Failed to load Coqui TTS:", str(e)[:100])
    except Exception as e:
        print("⚠️  Failed to load Coqui TTS:", str(e)[:100])

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
    # don't attempt generation on empty or whitespace-only prompt
    if not text or not text.strip():
        return None
    ensure_tts_loaded()
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
            audio = librosa.resample(audio, orig_sr=48000, target_sr=TARGET_SR)

        elif engine == "coqui":
            if coqui is None:
                return None
            kwargs = {"text": text}
            if coqui_language:
                kwargs["language"] = coqui_language
            if coqui_speaker:
                kwargs["speaker"] = coqui_speaker
            try:
                audio = coqui.tts(**kwargs)
            except TypeError:
                audio = coqui.tts(text)
            audio = np.array(audio)
            # try to get sample rate from model utils if available
            src_sr = 22050
            try:
                src_sr = getattr(coqui, "sample_rate", src_sr)
            except Exception:
                pass
            if src_sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=src_sr, target_sr=TARGET_SR)

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
                    safe_extract_tar(tar, tmpdir)

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
                for prompt_file in ("PROMPTS", "prompts.txt"):
                    prompts_path = os.path.join(etc_dir, prompt_file)
                    if os.path.exists(prompts_path):
                        with open(prompts_path, encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                parts = line.strip().split(" ", 1)
                                if len(parts) == 2:
                                    key = os.path.basename(parts[0])
                                    prompts[key] = parts[1]
                        break

                # если нет расшифровок / prompts пустой, пропускаем архив целиком
                if not prompts:
                    print(f"Skipping archive {link} – no transcripts found")
                    continue

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
