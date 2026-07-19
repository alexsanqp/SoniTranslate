#!/usr/bin/env python3
"""Direct StyleTTS2-UK dubbing (no server): load the model from local files,
synthesize each subtitle line on CPU, assemble with anchor sync, mux to video.

Replicates the text-normalization + synthesis chain from the tts-server
styletts2_uk provider, but loads weights from local paths (config.yml +
pytorch_model.bin) so it needs no HuggingFace access.

Usage:
  python styletts2_dub.py --srt uk.srt --config config.yml \
      --weights pytorch_model.bin --voice "Марина Панас.pt" \
      --video in.mp4 --output out.mp4
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from unicodedata import normalize as _uni

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # reuse cpu_dub helpers
import cpu_dub  # noqa: E402

# --- text normalization (ported from tts_server.providers.styletts2_uk) ----
_DASH_CHARS_RE = re.compile(r"[᠆‐‑‒–—―⁻₋−⸺⸻]")
_SPACED_HYPHEN_RE = re.compile(r" - ")
_NUMBER_RE = re.compile(r"(?:(?<=^)|(?<=[\s(\[]))-?\d+(?:[.,]\d+)?")


def _normalize_digits_uk(text):
    from num2words import num2words

    def _r(m):
        tok = m.group(0)
        for sep in (".", ","):
            if sep in tok:
                neg = tok.startswith("-")
                body = tok[1:] if neg else tok
                ip, fp = body.split(sep, 1)
                spoken = (f"{num2words(int(ip or '0'), lang='uk')} кома "
                          f"{num2words(int(fp), lang='uk')}")
                return f"мінус {spoken}" if neg else spoken
        return num2words(int(tok), lang="uk")

    return _NUMBER_RE.sub(_r, text)


_ACUTE = "́"  # combining acute accent = StressSymbol.CombiningAcuteAccent
# Spell out abbreviations so the phonemizer says them, not a consonant blob.
_ABBREV = {"ДППГ": "де-пе-пе-ге", "BPPV": "бі-пі-пі-ві"}


def _normalize_uk(text):
    for k, v in _ABBREV.items():
        text = text.replace(k, v)
    text = text.replace("+", _ACUTE)
    text = _uni("NFKC", text)
    # "30-60" -> "30 до 60" so both numbers get spoken (range reads naturally).
    text = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", " до ", text)
    text = _normalize_digits_uk(text)
    text = _DASH_CHARS_RE.sub("-", text)
    if not text:
        return text
    if text[-1] not in ".?!:-":
        text = text + "."
    text = _SPACED_HYPHEN_RE.sub(": ", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True, help="Ukrainian subtitles (timed)")
    ap.add_argument("--config", required=True, help="StyleTTS2 config.yml")
    ap.add_argument("--weights", required=True, help="StyleTTS2 pytorch_model.bin")
    ap.add_argument("--voice", required=True, help="voice style .pt file")
    ap.add_argument("--video", default=None, help="video to mux the dub over")
    ap.add_argument("--output", default="styletts2_dub.wav")
    ap.add_argument("--speed", type=float, default=1.0, help="model speaking speed")
    ap.add_argument("--max-speed", type=float, default=1.5, help="fit-to-slot cap")
    ap.add_argument("--sync", default="anchor", choices=["anchor", "sequential"])
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="st2_dub_")
    print(f"[st2] work dir: {workdir}", flush=True)

    import soundfile
    import torch
    import ukrainian_accentor as accentor
    from ipa_uk import ipa
    from styletts2_inference.models import StyleTTS2

    # Stress via the bundled neural accentor (no stanza, no network download).
    def stressify(t):
        return accentor.process(t, mode="stress")

    print("[st2] loading model on CPU ...", flush=True)
    model = StyleTTS2(config_path=args.config, weights_path=args.weights,
                      device="cpu")
    style = torch.load(args.voice, map_location="cpu", weights_only=True)
    print("[st2] model ready", flush=True)

    def synth(text, wav_path):
        normalized = _normalize_uk(text)
        stressed = stressify(normalized)
        phonemes = ipa(stressed)
        tokens = model.tokenizer.encode(phonemes)
        with torch.no_grad():
            wav = model(tokens, speed=args.speed, s_prev=style)
        wav_np = wav.cpu().numpy() if hasattr(wav, "cpu") else wav
        soundfile.write(wav_path, wav_np, 24000, format="WAV")

    segments = cpu_dub.parse_srt(args.srt)
    for s in segments:
        s["translated"] = s["text"]
    print(f"[st2] {len(segments)} segments", flush=True)

    track = cpu_dub.assemble_track(
        segments, synth, workdir, max_speed=args.max_speed, sync=args.sync
    )
    audio_wav = os.path.join(workdir, "dub.wav")
    track.export(audio_wav, format="wav")

    out = args.output
    if args.video and out.lower().endswith((".mp4", ".mkv", ".mov", ".webm")):
        print("[st2] muxing over video ...", flush=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", args.video, "-i", audio_wav,
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "160k", "-metadata:s:a:0", "language=ukr",
             "-shortest", out], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(["ffmpeg", "-y", "-i", audio_wav, out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[st2] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
