# Runa

A whisper made permanent. Real-time, fully on-device transcription for
Debian + GNOME, targeting AMD Ryzen 7040-series iGPUs (Radeon 780M) via
Vulkan.

```
mic ──► ring buffer ──► energy VAD gate ──► chunk slicer ──► whisper-cli ──► partial/final reconcile ──► stdout
        (fixed RAM)     (skip silence)      (1s/5s/10s)     (--vad model)
```

## Quick start

```bash
./setup.sh                      # deps, whisper.cpp (Vulkan), models
python3 runa.py --mode 5s --lang sv
```

Flags: `--mode 1s|5s|10s`, `--lang sv` (default; `auto` for detection),
`--model models/ggml-base.bin`, `--wav FILE` (offline validation, no mic
needed).

## Architecture

```
┌──────────────┐   16 kHz mono f32   ┌────────────────────┐
│ sounddevice  ├────────────────────►│ fixed ring buffer  │  deque(maxlen) — RAM
│ (PipeWire/   │  1024-sample frames │ hard cap 600 s     │  constant by construct
│  PulseAudio) │                     └─────────┬──────────┘
└──────────────┘                               │ per frame
                                     ┌─────────▼──────────┐
                                     │ energy VAD gate    │  RMS threshold + quiet-run
                                     │ (cheap, in Python) │  endpoint detection (hang)
                                     └─────────┬──────────┘
                          voice                │ silence → drop
                        ┌──────────────────────┴─────────────┐
                        │ chunk slicer                        │
                        │  mode 1s: step 1 s, window 6 s      │
                        │  mode 5s: step 5 s, window 10 s     │
                        │  mode 10s: step 10 s, window 15 s   │
                        └──────────────────┬──────────────────┘
                                           │ temp WAV
                        ┌──────────────────▼──────────────────┐
                        │ whisper-cli subprocess              │
                        │  --vad --vad-model ggml-silero      │
                        │  → precise speech boundaries        │
                        └──────────────────┬──────────────────┘
                                           │ text
                        ┌──────────────────▼──────────────────┐
                        │ partial / final reconciliation      │
                        │  partials stream from live window   │
                        │  final overwrites pending partials  │
                        │  on VAD endpoint                    │
                        └─────────────────────────────────────┘
```

## Apple SpeechAnalyzer comparison

| Aspect        | Apple SpeechAnalyzer        | Runa                             |
|---------------|------------------------------|----------------------------------|
| On-device     | Yes (Apple Silicon ANE)     | Yes (Vulkan on 780M iGPU, or CPU AVX2) |
| Streaming     | Chunked with endpointing    | Same: chunked windows + VAD endpointing |
| Partial/final | Streaming partials, finalized on endpoint | Same reconciliation model |
| VAD           | Neural (built-in)           | Energy gate (Python) + silero v5.1.2 inside whisper-cli for precise boundaries |
| Acceleration  | Neural Engine               | Vulkan compute (GGML_VULKAN=1); no NPU path on AMD Linux, iGPU is the closest analogue |
| Platform      | macOS/iOS                   | Debian GNU/Linux, GNOME/Wayland  |

## Performance (measured on hub, Ryzen mobile-class, CPU build)

- RTF (real-time factor, lower = better) for `base` model, AVX2 CPU,
  16 threads: **0.03** on 20 s audio. Every mode is comfortably real-time
  on CPU; Vulkan lowers latency further and frees CPU cores.
- Expected end-to-end latency per mode (approx, base model):

  | Mode | Partial period | Final latency | Use case |
  |------|----------------|----------------|----------|
  | 1s   | ~1 s           | ~1-2 s after speech ends | captions, live dictation |
  | 5s   | ~5 s           | ~2-4 s | default, balanced |
  | 10s  | ~10 s          | ~3-6 s | best accuracy, meeting notes |

  Latency budget per stage in `docs/DESIGN.md`.

## Memory discipline

- Ring buffer: `collections.deque(maxlen=...)` — appending an element to a
  full deque silently drops the oldest; RAM is O(1) in session length.
- Hard cap assertion: the buffer can never exceed
  `--max-buffer-seconds` (default 600 s); the code raises if it ever could.
- whisper.cpp context is fixed at load time (base model ≈ 500 MB RAM,
  well within 16 GB).
- Measured RSS for a 30 s offline run: ~190 MB Python+numpy total,
  ~36 MB with the VAD gate rejecting all silence (whisper never spawned).

## Build notes

- Primary target: `GGML_VULKAN=1` (Radeon 780M via RADV). If `vulkaninfo`
  finds no ICD or the build fails, `setup.sh` falls back to a native CPU
  (AVX2) build and records it in `BUILD-MODE.txt`.
- `setup.sh` is idempotent: re-running re-uses the whisper.cpp clone,
  skips existing models, and only installs missing packages.

## Roadmap

- **Round 2:** GTK4/libadwaita GUI — live waveform, partial text with
  fade-to-final, push-to-talk, model picker.
- **Round 3:** Flatpak packaging (org.freedesktop.Platform + pipewire
  portal), Flathub release.
- Later: speaker diarization, word-level timestamps UI, WhisperVulkan
  tuning for RDNA3 (wave64), streaming websocket server mode.

## Links

Part of the [vhsgreed](https://vhsgreed.win) toolset: data, code, and methods in the open.
