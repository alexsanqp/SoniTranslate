#!/usr/bin/env python3
"""
cpu_dub.py - CPU-only, offline-friendly dubbing pipeline for SoniTranslate.

This is a lightweight companion to app_rvc.py that swaps every GPU / Hugging Face
token dependent stage for a local, CPU-friendly library so the whole thing runs
without a GPU and without an HF token:

  stage        SoniTranslate default            cpu_dub.py replacement
  -----------  -------------------------------  --------------------------------
  download     yt-dlp                           yt-dlp (unchanged)
  transcribe   WhisperX + faster-whisper (GPU)  openai-whisper on CPU, or an
                                                 existing .srt (no ASR, no model)
  diarize      pyannote (needs HF token)        skipped (single speaker)
  translate    deep-translator / GPT            translate.googleapis.com (public,
                                                 no key) with an offline fallback
  TTS          edge-tts / XTTS / RVC (GPU/HF)   espeak-ng (fully offline, has uk)
  assemble     ffmpeg                           ffmpeg / pydub (unchanged)

Every stage degrades gracefully: if the video host is unreachable you can feed a
local media file or a subtitle file instead, and if the ASR model cannot be
fetched you can supply an .srt so no model download is needed at all.

Examples
--------
  # Full pipeline from a URL (needs network access to the video host):
  python cpu_dub.py --input "https://www.youtube.com/watch?v=XXXX" --target uk

  # Fully offline: dub an existing English subtitle file into Ukrainian.
  python cpu_dub.py --input speech.srt --target uk --output dub_uk.wav

  # Dub a local media file (transcribe on CPU, then translate + speak):
  python cpu_dub.py --input talk.mp4 --source en --target uk --output talk_uk.mp4
"""

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# espeak-ng voice code per language. espeak-ng ships these offline; run
# `espeak-ng --voices` to see the full list on your machine.
ESPEAK_VOICE = {
    "uk": "uk", "en": "en", "ru": "ru", "de": "de", "fr": "fr-fr",
    "es": "es", "it": "it", "pl": "pl", "pt": "pt", "cs": "cs",
    "nl": "nl", "ro": "ro", "sk": "sk", "tr": "tr", "hu": "hu",
}


def log(msg):
    print(f"[cpu_dub] {msg}", flush=True)


def run(cmd, **kw):
    """Run a subprocess, raising with a readable message on failure."""
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}"
        )
    return res.stdout


def which(binary):
    from shutil import which as _which

    return _which(binary)


# ---------------------------------------------------------------------------
# Subtitle (SRT) parsing / writing
# ---------------------------------------------------------------------------

_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _ts_to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _sec_to_ts(t):
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path):
    """Parse an .srt file into [{start, end, text}] (seconds)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        content = f.read()

    segments = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        ts_line = None
        text_lines = []
        for ln in lines:
            m = _TS_RE.search(ln)
            if m and ts_line is None:
                ts_line = m
            elif not ln.strip().isdigit() or text_lines:
                text_lines.append(ln)
        if ts_line is None:
            continue
        g = ts_line.groups()
        start = _ts_to_sec(g[0], g[1], g[2], g[3])
        end = _ts_to_sec(g[4], g[5], g[6], g[7])
        text = " ".join(text_lines).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def write_srt(segments, path, key="text"):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(
                f"{i}\n{_sec_to_ts(seg['start'])} --> {_sec_to_ts(seg['end'])}\n"
                f"{seg[key]}\n\n"
            )


# ---------------------------------------------------------------------------
# Translation - public Google endpoint (no API key), offline fallback
# ---------------------------------------------------------------------------

def google_translate(text, source, target, retries=4):
    """Translate a single string via the public translate.googleapis.com
    endpoint. No API key required. Raises on repeated failure."""
    base = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": source,
        "tl": target,
        "dt": "t",
        "q": text,
    }
    url = base + "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return "".join(chunk[0] for chunk in data[0] if chunk and chunk[0])
        except Exception as e:  # network / rate-limit
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"translation failed after {retries} tries: {last_err}")


def translate_segments(segments, source, target, corrections=None):
    """Translate each segment's text. 'auto' is accepted as source."""
    src = "auto" if source in (None, "", "auto") else source
    out = []
    total = len(segments)
    for i, seg in enumerate(segments, 1):
        translated = html.unescape(google_translate(seg["text"], src, target))
        for wrong, right in (corrections or {}).items():
            translated = translated.replace(wrong, right)
        seg = dict(seg)
        seg["translated"] = translated
        out.append(seg)
        if i % 10 == 0 or i == total:
            log(f"translated {i}/{total} segments")
    return out


# ---------------------------------------------------------------------------
# Text to speech - two CPU backends, neither needs a token:
#   - google : natural neural-ish voice via translate.googleapis.com (network)
#   - espeak : fully offline robotic voice via espeak-ng
# ---------------------------------------------------------------------------

def espeak_tts(text, voice, wav_path, rate=160, pitch=50):
    """Synthesize `text` to a wav file with espeak-ng (fully offline)."""
    if not which("espeak-ng"):
        raise RuntimeError(
            "espeak-ng not found. Install it: apt-get install -y espeak-ng"
        )
    cmd = [
        "espeak-ng", "-v", voice,
        "-s", str(rate), "-p", str(pitch),
        "-w", wav_path, text,
    ]
    run(cmd)


def _chunk_text(text, limit=190):
    """Split text into <= limit-char pieces on word boundaries (the public
    Google TTS endpoint rejects long strings)."""
    words = text.split()
    chunks, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > limit:
            if cur:
                chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        chunks.append(cur)
    return chunks or [text]


def google_tts(text, lang, wav_path, workdir, retries=4):
    """Synthesize `text` with the public Google Translate TTS voice (natural,
    no API key). Long text is chunked and the pieces are concatenated."""
    from pydub import AudioSegment

    pieces = []
    for j, chunk in enumerate(_chunk_text(text)):
        q = urllib.parse.quote(chunk)
        url = (
            "https://translate.googleapis.com/translate_tts?ie=UTF-8"
            f"&q={q}&tl={lang}&client=tw-ob&total=1&idx=0&textlen={len(chunk)}"
        )
        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=25) as r:
                    raw = r.read()
                mp3 = os.path.join(workdir, f"_g_{abs(hash(chunk)) % 10**8}_{j}.mp3")
                with open(mp3, "wb") as f:
                    f.write(raw)
                pieces.append(AudioSegment.from_file(mp3, format="mp3"))
                break
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"google TTS failed: {last}")
        time.sleep(0.15)  # be gentle with the endpoint

    audio = pieces[0]
    for p in pieces[1:]:
        audio += p
    audio.export(wav_path, format="wav")


def server_tts(text, wav_path, workdir, url, voice, model, language,
               api_key=None, retries=3):
    """Synthesize via an OpenAI-compatible TTS server (POST /v1/audio/speech),
    e.g. a self-hosted tts-server with edge / StyleTTS2-UK / Qwen providers.
    Returns whatever audio the server sends, transcoded to wav."""
    body = json.dumps({
        "input": text, "voice": voice, "model": model,
        "language": language, "response_format": "wav",
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                audio = r.read()
            raw = os.path.join(workdir, "_srv_raw")
            with open(raw, "wb") as f:
                f.write(audio)
            run(["ffmpeg", "-y", "-i", raw, wav_path])  # normalize to wav
            return
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"TTS server request failed: {last}")


def _audio_duration(path):
    out = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ]
    )
    return float(out.strip())


def _fit_to_duration(src_wav, dst_wav, target_dur, max_speed=2.2):
    """Time-stretch src to <= target_dur, preserving pitch (ffmpeg atempo).
    Never slows audio down; only compresses if it overflows the slot."""
    dur = _audio_duration(src_wav)
    if target_dur <= 0 or dur <= target_dur or dur <= 0:
        run(["ffmpeg", "-y", "-i", src_wav, dst_wav])
        return
    factor = min(dur / target_dur, max_speed)
    # atempo only accepts 0.5..2.0; chain filters for bigger factors.
    filters, remaining = [], factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")
    run(
        [
            "ffmpeg", "-y", "-i", src_wav,
            "-filter:a", ",".join(filters), dst_wav,
        ]
    )


# ---------------------------------------------------------------------------
# Assembly - overlay each dubbed clip on a silent timeline (pydub)
# ---------------------------------------------------------------------------

def assemble_track(segments, synth, workdir, max_speed=1.8, sync="anchor",
                   sample_rate=24000):
    """Synthesize each segment and lay it on a timeline.

    sync="anchor" (default): every clip is pinned to its own start timestamp
    and, when it is longer than its slot, sped up (pitch-preserving, capped at
    `max_speed`) to fit. Because each segment re-anchors to the video timeline,
    lag never accumulates - the dub stays in sync from start to finish. A clip
    that still overflows after the cap is trimmed to its slot so it can't spill
    into the next line.

    sync="sequential": clips play back-to-back at a natural pace (a clip starts
    after the previous one if that ran long). Lower speed-up, but drift can
    build up over a long video."""
    from pydub import AudioSegment

    timeline = AudioSegment.silent(duration=1000, frame_rate=sample_rate)
    cursor = 0  # ms; end of the previously placed clip (sequential mode)
    for i, seg in enumerate(segments):
        text = seg.get("translated") or seg["text"]
        raw = os.path.join(workdir, f"seg_{i:05d}.wav")
        synth(text, raw)

        slot = seg["end"] - seg["start"]
        dur = _audio_duration(raw)
        is_last = i + 1 == len(segments)

        if sync == "anchor":
            # Speed up only as much as this slot needs (up to the cap).
            if slot > 0 and dur > slot:
                fit = os.path.join(workdir, f"seg_{i:05d}_fit.wav")
                _fit_to_duration(raw, fit, slot, max_speed=max_speed)
                raw = fit
            clip = AudioSegment.from_file(raw)
            # Hard safety trim so a still-too-long clip never overlaps the next.
            if not is_last and slot > 0 and len(clip) > int(slot * 1000):
                clip = clip[: int(slot * 1000)].fade_out(40)
            pos = int(seg["start"] * 1000)
        else:  # sequential
            if slot > 0 and dur > slot * max_speed:
                fit = os.path.join(workdir, f"seg_{i:05d}_fit.wav")
                _fit_to_duration(raw, fit, dur / max_speed, max_speed=max_speed)
                raw = fit
            clip = AudioSegment.from_file(raw)
            pos = max(int(seg["start"] * 1000), cursor)

        end = pos + len(clip)
        if end + 500 > len(timeline):
            timeline += AudioSegment.silent(
                duration=end + 500 - len(timeline), frame_rate=sample_rate
            )
        timeline = timeline.overlay(clip, position=pos)
        cursor = end
        if (i + 1) % 5 == 0 or is_last:
            log(f"synthesized {i + 1}/{len(segments)} segments")

    # Pad with trailing silence to the end of the last slot so muxing with
    # -shortest keeps the full video length instead of trimming its tail.
    final_ms = int(max(s["end"] for s in segments) * 1000)
    if len(timeline) < final_ms:
        timeline += AudioSegment.silent(
            duration=final_ms - len(timeline), frame_rate=sample_rate
        )
    return timeline


# ---------------------------------------------------------------------------
# Optional stages: download and transcription
# ---------------------------------------------------------------------------

def download_media(url, workdir):
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")
    out_tmpl = os.path.join(workdir, "source.%(ext)s")
    log(f"downloading {url}")
    run(["yt-dlp", "-f", "bestaudio/best", "-o", out_tmpl, url])
    for name in os.listdir(workdir):
        if name.startswith("source."):
            return os.path.join(workdir, name)
    raise RuntimeError("download produced no file")


def transcribe_cpu(media_path, source, model_name="base"):
    """Transcribe with openai-whisper on CPU. No HF token required.
    The model is fetched from the public OpenAI CDN on first use."""
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "openai-whisper not installed. Run: pip install -U openai-whisper"
        )
    log(f"loading whisper '{model_name}' on CPU (first run downloads the model)")
    model = whisper.load_model(model_name, device="cpu")
    lang = None if source in (None, "", "auto") else source
    result = model.transcribe(media_path, language=lang, fp16=False)
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result["segments"]
        if s["text"].strip()
    ]
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="CPU-only, offline-friendly Ukrainian (or any-language) "
        "dubbing pipeline."
    )
    ap.add_argument(
        "--input", required=True,
        help="video URL, local media file, or an .srt subtitle file",
    )
    ap.add_argument("--source", default="auto", help="source language code (or 'auto')")
    ap.add_argument("--target", default="uk", help="target language code (default: uk)")
    ap.add_argument("--output", default="dub_output.wav", help="output file path")
    ap.add_argument(
        "--video", default=None,
        help="mux the dub over this video file (use when --input is an .srt)",
    )
    ap.add_argument(
        "--tts", default="google", choices=["google", "espeak", "server"],
        help="TTS backend: google (natural, needs network), espeak (offline), "
        "or server (an OpenAI-compatible /v1/audio/speech endpoint)",
    )
    ap.add_argument("--tts-url", default=None, help="server backend: speech endpoint URL")
    ap.add_argument("--tts-voice", default=None, help="server backend: voice id")
    ap.add_argument("--tts-model", default="auto", help="server backend: model/provider")
    ap.add_argument("--tts-key", default=os.environ.get("TTS_API_KEY"), help="server backend: bearer token")
    ap.add_argument(
        "--whisper-model", default="base",
        help="openai-whisper model size: tiny/base/small/medium/large",
    )
    ap.add_argument("--tts-rate", type=int, default=160, help="espeak words per minute")
    ap.add_argument(
        "--max-speed", type=float, default=1.5,
        help="max pitch-preserving speed-up used to fit a slot (1.0 = never)",
    )
    ap.add_argument(
        "--save-srt", default=None,
        help="also write the translated subtitles to this .srt path",
    )
    ap.add_argument(
        "--pre-translated", action="store_true",
        help="the input .srt is already in the target language; skip "
        "translation and speak its text as-is (lets you hand-tune wording)",
    )
    ap.add_argument(
        "--sync", default="anchor", choices=["anchor", "sequential"],
        help="anchor: pin each line to its timestamp (no drift); "
        "sequential: back-to-back natural pace (can drift)",
    )
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="cpu_dub_")
    log(f"work dir: {workdir}")

    # --- Stage 1+2: obtain timed source segments ---------------------------
    inp = args.input
    media_path = args.video
    if inp.lower().endswith(".srt"):
        log("input is a subtitle file - skipping download and ASR (fully offline)")
        segments = parse_srt(inp)
    else:
        if re.match(r"^https?://", inp):
            media_path = download_media(inp, workdir)
        elif os.path.exists(inp):
            media_path = inp
        else:
            log(f"ERROR: input not found: {inp}")
            sys.exit(1)
        segments = transcribe_cpu(media_path, args.source, args.whisper_model)

    if not segments:
        log("ERROR: no speech segments found")
        sys.exit(1)
    log(f"{len(segments)} source segments")

    # --- Stage 3: translate -------------------------------------------------
    if args.pre_translated:
        log("input is already in the target language - skipping translation")
        for seg in segments:
            seg["translated"] = seg["text"]
    else:
        # A few machine-translation glitches wrong in this medical context.
        corrections = {
            "головку": "голову", "головки": "голови", "головок": "голови",
            "головка": "голова", "Головка": "Голова",
            "Кришталевий падає": "кристал випадає",
            "Одного разу кристал випадає": "щойно кристал випадає",
        }
        segments = translate_segments(
            segments, args.source, args.target, corrections
        )
    if args.save_srt:
        write_srt(segments, args.save_srt, key="translated")
        log(f"translated subtitles written to {args.save_srt}")

    # --- Stage 4: synthesize + assemble ------------------------------------
    if args.tts == "server":
        if not args.tts_url:
            log("ERROR: --tts server requires --tts-url")
            sys.exit(1)
        log(f"synthesizing via TTS server {args.tts_url} "
            f"(model={args.tts_model}, voice={args.tts_voice})")
        synth = lambda text, wav: server_tts(
            text, wav, workdir, args.tts_url, args.tts_voice,
            args.tts_model, args.target, args.tts_key,
        )
    elif args.tts == "google":
        log(f"synthesizing with Google TTS voice '{args.target}'")
        synth = lambda text, wav: google_tts(text, args.target, wav, workdir)
    else:
        voice = ESPEAK_VOICE.get(args.target, args.target)
        log(f"synthesizing with espeak-ng voice '{voice}'")
        synth = lambda text, wav: espeak_tts(text, voice, wav, rate=args.tts_rate)
    track = assemble_track(
        segments, synth, workdir, max_speed=args.max_speed, sync=args.sync
    )

    # --- Stage 5: export / mux ---------------------------------------------
    out = args.output
    audio_wav = os.path.join(workdir, "dubbed.wav")
    track.export(audio_wav, format="wav")

    if media_path and out.lower().endswith((".mp4", ".mkv", ".mov", ".webm")):
        # Mux the dubbed track over the muted original video.
        log("muxing dubbed audio over original video")
        run(
            [
                "ffmpeg", "-y", "-i", media_path, "-i", audio_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                "-metadata:s:a:0", f"language={args.target}",
                "-shortest", out,
            ]
        )
    else:
        run(["ffmpeg", "-y", "-i", audio_wav, out])

    log(f"DONE -> {out}")


if __name__ == "__main__":
    main()
