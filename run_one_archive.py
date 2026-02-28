#!/usr/bin/env python3
import os
import tarfile
import tempfile
from main import get_archive_links, download_file, process_audio, generate_tts, engines, OUT_REAL, OUT_FAKE, TARGET_SR
import soundfile as sf


def run_one(limit_utts=5):
    links = get_archive_links()
    if not links:
        print("No archive links found")
        return

    # find first archive that contains any transcripts (look for PROMPTS or prompts.txt)
    link = None
    for l in links[:10]:  # only check first few archives to keep the test quick
        print("Checking archive for transcripts:", l)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tmp.tgz")
            download_file(l, path)
            with tarfile.open(path) as tar:
                tar.extractall(tmpdir)
            extracted = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d)) and d != "__MACOSX"]
            if not extracted:
                continue
            etc = os.path.join(tmpdir, extracted[0], "etc")
            prompts = {}
            # check both uppercase and lowercase prompt filenames
            for fname in ["PROMPTS", "prompts.txt"]:
                pfile = os.path.join(etc, fname)
                if os.path.exists(pfile):
                    with open(pfile, encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2:
                                key = os.path.basename(parts[0])
                                prompts[key] = parts[1]
                    break
            if prompts:
                link = l
                print("Selected archive with transcripts:", link)
                break
    if link is None:
        print("No archive with transcripts found")
        return

    utt_counter = 0
    metadata = []

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "data.tgz")
        download_file(link, archive_path)

        with tarfile.open(archive_path) as tar:
            tar.extractall(tmpdir)

        extracted = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d)) and d != "__MACOSX"]
        if not extracted:
            print("No extracted speaker dir")
            return

        speaker_id = extracted[0]
        wav_dir = os.path.join(tmpdir, speaker_id, "wav")
        etc_dir = os.path.join(tmpdir, speaker_id, "etc")

        prompts = {}
        # support both PROMPTS and prompts.txt
        for fname in ["PROMPTS", "prompts.txt"]:
            prompts_path = os.path.join(etc_dir, fname)
            if os.path.exists(prompts_path):
                with open(prompts_path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            # normalize the key the same way the main pipeline does
                            key = os.path.basename(parts[0])
                            prompts[key] = parts[1]
                break

        if not prompts:
            print("No transcripts in archive, skipping")
            return

        os.makedirs(OUT_REAL, exist_ok=True)

        for wav_file in os.listdir(wav_dir):
            if not wav_file.lower().endswith('.wav'):
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

            for engine in engines:
                fake_audio = generate_tts(engine, text)
                if fake_audio is None:
                    continue
                fake_out = os.path.join(OUT_FAKE, engine, f"{utt_id}.wav")
                os.makedirs(os.path.dirname(fake_out), exist_ok=True)
                sf.write(fake_out, fake_audio, TARGET_SR, subtype="PCM_16")

            metadata.append({"utt_id": utt_id, "speaker": speaker_id, "text": text})

            utt_counter += 1
            if utt_counter >= limit_utts:
                break

    if metadata:
        import pandas as pd
        df = pd.DataFrame(metadata)
        df.to_csv("metadata_one_archive.csv", index=False)

    print("Done. Processed:", utt_counter)


if __name__ == '__main__':
    run_one(limit_utts=5)
