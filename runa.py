#!/usr/bin/env python3
"""Runa: real-time on-device transcription (whisper.cpp + Vulkan/CPU).

Pipeline:
  mic (sounddevice, 16 kHz mono f32)
    -> fixed ring buffer (deque maxlen; RAM is O(1) in session length)
    -> energy VAD gate (skip silence, keep a pre-roll tail)
    -> chunk slicer by mode (1s / 5s / 10s)
    -> whisper-cli subprocess per segment (--vad --vad-model)
    -> partial/final reconciliation (Apple SpeechAnalyzer style):
         partials stream from the live window, a final replaces the
         pending partial once the VAD declares an endpoint.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

SR = 16000
SAMPLE_DTYPE = np.float32

# Fixed memory cap: the ring buffer never holds more than this many seconds.
MAX_BUFFER_SECONDS = 600  # generous safety cap; modes use far less


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def find_whisper_cli() -> str:
    env = os.environ.get("RUNA_WHISPER_CLI")
    if env and os.path.exists(env):
        return env
    root = repo_root()
    for cand in (
        os.path.join(root, "whisper.cpp", "build-vulkan", "bin", "whisper-cli"),
        os.path.join(root, "whisper.cpp", "build", "bin", "whisper-cli"),
        os.path.join(root, "whisper.cpp", "build", "whisper-cli"),
    ):
        if os.path.exists(cand):
            return cand
    found = shutil.which("whisper-cli")
    if found:
        return found
    raise FileNotFoundError(
        "whisper-cli not found. Run ./setup.sh first, or set RUNA_WHISPER_CLI."
    )


def default_model() -> str:
    env = os.environ.get("RUNA_MODEL")
    if env:
        return env
    p = os.path.join(repo_root(), "models", "ggml-base.bin")
    return p if os.path.exists(p) else p  # fail loudly later with a clear msg


def default_vad_model() -> str:
    return os.environ.get(
        "RUNA_VAD_MODEL",
        os.path.join(repo_root(), "models", "ggml-silero-v5.1.2.bin"),
    )


# ---------------------------------------------------------------- VAD gate
class EnergyVAD:
    """Cheap energy gate to skip feeding silence to whisper.

    whisper-cli --vad does the precise boundary work; this gate only decides
    whether a chunk is worth sending at all, and when a voice 'endpoint'
    (quiet run) has occurred so we can finalize.
    """

    def __init__(self, threshold: float = 0.004, hang_frames: int = 12):
        # hang_frames * window_ms below threshold => endpoint
        self.threshold = threshold
        self.quiet_run = 0
        self.hang_frames = hang_frames

    def is_voice(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        return rms >= self.threshold

    def update(self, chunk: np.ndarray, window_seconds: float) -> bool:
        """Returns True when an endpoint (finalize) should fire."""
        if self.is_voice(chunk):
            self.quiet_run = 0
            return False
        self.quiet_run += 1
        hang = max(1, int(round(self.hang_frames * (window_seconds / 0.032))))
        return self.quiet_run >= hang


# ------------------------------------------------------------ segment store
@dataclass
class Segment:
    samples: deque  # of np.ndarray windows
    started_at: float


class RingBuffer:
    """Fixed-capacity audio ring. RAM use is constant by construction."""

    def __init__(self, max_seconds: int = MAX_BUFFER_SECONDS):
        cap = int(max_seconds * SR)
        self._buf: deque[np.ndarray] = deque(maxlen=cap // 1024)  # 1024-sample frames
        self._len = 0

    def push(self, chunk: np.ndarray) -> None:
        self._buf.append(chunk)
        self._len = min(self._len + len(chunk), self._buf.maxlen * 1024)

    def extend(self, chunks) -> None:
        for c in chunks:
            self.push(c)

    def __len__(self) -> int:
        return self._len

    def snapshot(self) -> np.ndarray:
        return np.concatenate(tuple(self._buf)) if self._buf else np.zeros(0, dtype=SAMPLE_DTYPE)


class Transcriber:
    def __init__(self, args):
        self.args = args
        self.cli = find_whisper_cli()
        self.vad = EnergyVAD()
        self.ring = RingBuffer()
        self.pending_final = ""  # last text awaiting reconciliation

    def run_whisper(self, wav_path: str) -> str:
        cmd = [
            self.cli,
            "-m", self.args.model,
            "-f", wav_path,
            "-l", self.args.lang,
            "--vad", "--vad-model", self.args.vad_model,
            "-np", "-t", str(self.args.threads),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return ""
        out = []
        for line in (r.stdout or "").splitlines():
            # whisper-cli lines look like: [00:00:00.000 --> 00:00:02.040] text
            if "]" in line:
                line = line.split("]", 1)[1].strip()
            if line:
                out.append(line)
        return " ".join(out).strip()

    def transcribe_window(self, samples: np.ndarray, final: bool) -> str | None:
        """Transcribe `samples`; returns final text (replacing partials) or None."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            import wave

            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(tmp, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(pcm.tobytes())
            text = self.run_whisper(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if final:
            return text
        return text or None  # partial (may be empty -> no update)


MODES = {
    "1s": {"step": 1.0, "window": 6.0},
    "5s": {"step": 5.0, "window": 10.0},
    "10s": {"step": 10.0, "window": 15.0},
}


def ts() -> str:
    return time.strftime("%H:%M:%S")


def run_mic(args):
    import sounddevice as sd

    mode = MODES[args.mode]
    step, window = mode["step"], mode["window"]
    frame = 1024  # samples per callback
    ring = RingBuffer(args.max_buffer_seconds)
    vad = EnergyVAD()
    pending = ""           # current partial shown
    speech: deque = deque()  # windows belonging to the open segment
    voice_seen = False
    until_next_emit = step
    t0 = time.time()

    def on_chunk(chunk: np.ndarray):
        nonlocal pending, voice_seen, until_next_emit
        ring.push(chunk)
        voiced = vad.is_voice(chunk)
        endpoint = vad.update(chunk, frame / SR)
        if voiced:
            speech.append(chunk)
            voice_seen = True
            until_next_emit -= len(chunk) / SR
        elif speech:
            # trailing tail keeps word boundaries; add then maybe finalize
            speech.append(chunk)
            until_next_emit -= len(chunk) / SR

        if voice_seen and (endpoint or until_next_emit <= 0 or len(speech) * frame >= SR * window):
            samples = np.concatenate(tuple(speech))
            speech.clear()
            voice_seen = False
            vad.quiet_run = 0
            until_next_emit = step
            final = bool(endpoint)
            text = tr.transcribe_window(samples, final=final)
            if final:
                if text:
                    print(f"[{ts()}] FINAL   {text}", flush=True)
                pending = ""
            elif text:
                pending = text
                print(f"[{ts()}] partial {pending}\r", end="", flush=True)

    tr = Transcriber(args)
    print(f"Runa | mode {args.mode} | lang {args.lang} | model {args.model}", flush=True)
    print("Listening... Ctrl-C to stop.", flush=True)
    with sd.InputStream(
        samplerate=SR, channels=1, dtype=SAMPLE_DTYPE, blocksize=frame,
        callback=lambda indata, frames, t, status: on_chunk(indata[:, 0].copy()),
    ):
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        while True:
            time.sleep(1)
            if len(ring) > MAX_BUFFER_SECONDS * SR:
                raise RuntimeError("ring buffer exceeded hard cap")  # cannot happen


def run_wav(args):
    """Offline validation: slice a WAV through the same VAD/chunk pipeline."""
    import wave

    with wave.open(args.wav, "rb") as w:
        assert w.getframerate() == SR, f"need {SR} Hz"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    samples = pcm.astype(SAMPLE_DTYPE) / 32768.0
    mode = MODES[args.mode]
    step = int(mode["step"] * SR)
    win = int(mode["window"] * SR)
    tr = Transcriber(args)
    vad = EnergyVAD()
    n = 0
    rtf_audio = 0.0
    t_start = time.time()
    for off in range(0, len(samples), step):
        chunk = samples[off: off + step]
        if not vad.is_voice(chunk) and not vad.quiet_run:
            continue
        seg = samples[max(0, off - SR // 2): min(len(samples), off + step + SR // 4)]
        if len(seg) < SR // 4:
            continue
        seg = np.pad(seg[:win], (0, max(0, win - len(seg[:win]))))[:win]
        n += 1
        text = tr.transcribe_window(seg, final=False)
        if text:
            print(f"[{ts()}] seg#{n:03d} {text}", flush=True)
        rtf_audio += len(seg) / SR
    wall = time.time() - t_start
    print(f"segments={n} audio={rtf_audio:.1f}s wall={wall:.1f}s "
          f"RTF={wall / rtf_audio if rtf_audio else float('nan'):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="runa", description=__doc__)
    ap.add_argument("--mode", choices=MODES.keys(), default="5s",
                    help="chunk step: 1s lowest latency, 10s best accuracy")
    ap.add_argument("--lang", default="sv", help="language code (or 'auto')")
    ap.add_argument("--model", default=default_model(), help="path to ggml model")
    ap.add_argument("--vad-model", default=default_vad_model())
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--max-buffer-seconds", type=int, default=MAX_BUFFER_SECONDS)
    ap.add_argument("--wav", help="offline mode: transcribe this WAV instead of the mic")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model} (run ./setup.sh)")
    if args.wav:
        run_wav(args)
    else:
        try:
            import sounddevice  # noqa: F401
        except OSError as e:
            sys.exit(f"no audio backend available: {e}\n"
                     "On Debian GNOME: sudo apt install libportaudio2 (setup.sh covers it), "
                     "or use --wav FILE for offline mode.")
        run_mic(args)


if __name__ == "__main__":
    main()
