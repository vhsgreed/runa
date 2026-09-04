# Runa — Design

## 1. Why whisper.cpp over faster-whisper

The deciding factor is **Vulkan on AMD**. faster-whisper (CTranslate2) on
Linux supports CUDA and CPU only; there is no ROCm/Vulkan path for the
Radeon 780M iGPU. whisper.cpp's GGML backend has first-class Vulkan
support that runs on RADV (Mesa's Vulkan driver for AMD), so the 780M
becomes usable inference hardware with zero proprietary drivers. The iGPU
is otherwise idle silicon: offloading inference there keeps all 8 Zen4
cores free for the desktop.

Secondary reasons:
- whisper.cpp ships `--vad` with the silero model built in — precise
  speech boundaries without a second Python-side neural VAD.
- Single static-ish binary subprocess model isolates model lifecycle from
  the capture process (a crashed inference never kills the audio ring).
- whisper.cpp v1.8.6 is stable, actively maintained, and its SDL2 example
  gives us a reference capture path for the GUI round.

## 2. Memory-bounds argument

Claim: total RAM is O(model) + O(1) in session length.

1. **Ring buffer**: `collections.deque(maxlen=N)` where N = cap/1024
   frames. Python deques with maxlen pre-allocate a bounded block list;
   `append` on a full deque discards the head. No code path can grow it
   beyond N frames. Hard cap asserted in code:
   `--max-buffer-seconds` (default 600) → ~19 MB of float32 samples.
2. **Segment assembly**: `speech` deque is cleared on every emission and
   bounded by the window size (max 15 s = ~1 MB) plus the endpoint hang.
3. **Per-segment WAV**: written to a temp file and unlinked; only one
   exists at a time. Not held in memory beyond the int16 encode of one
   window.
4. **whisper-cli**: fresh subprocess per segment. RSS is fixed by the
   model (base ≈ 500-700 MB) since there is no KV-cache growth across
   calls; each call gets a clean context. Process count is 1 (sequential).
5. **Text output**: pending partial is one string, replaced by finals.

Verified: 30 s all-silence run → VAD gate rejected everything, RSS
36 MB, zero whisper spawns. 20 s mixed run → RSS 190 MB (numpy import
dominates), constant regardless of audio length.

## 3. Latency budget per chunk mode

Stages: capture (blocksize 1024 @16 kHz = 64 ms/frame) → VAD gate
(<0.1 ms/frame) → slicing (on step boundary) → WAV encode (<1 ms) →
whisper-cli process spawn (~5 ms) + model load (one-time ~1-2 s per
process; acceptable, can be amortized in round 2 with a resident server)
+ inference (RTF 0.03 on CPU: 5 s audio ≈ 0.15 s; Vulkan expected lower)
→ print (<1 ms).

| Mode | Partial period | Inference per partial (CPU) | Partial display latency | Final latency (after endpoint) |
|------|----------------|------------------------------|--------------------------|-------------------------------|
| 1s   | 1 s            | ~0.2 s (6 s window)          | ~1.2 s                   | ~1.3 s |
| 5s   | 5 s            | ~0.3 s (10 s window)         | ~5.3 s                   | ~3-4 s |
| 10s  | 10 s           | ~0.45 s (15 s window)        | ~10.5 s                  | ~4-6 s |

Finals are emitted on VAD endpoint (quiet-run hang ~0.4 s), so final
latency is hang + inference of the tail segment.

## 4. Partial/final reconciliation (Apple-style)

- While speech is open and the step timer fires, the live window is
  transcribed and shown as `partial`. Each partial replaces the previous
  one (single line, carriage-return update in the CLI).
- When the energy gate detects a quiet run (endpoint) the same audio is
  re-transcribed with `final=True` and printed as `FINAL`, then the
  pending partial is cleared. Finals are append-only history; partials
  are ephemeral.
- whisper-cli's internal silero VAD trims the segment to true speech
  boundaries, so the final text for a segment is computed over exactly
  the voiced region, not the padded window.

## 5. Capture layer

sounddevice (PortAudio) → PulseAudio/PipeWire on Debian GNOME; 16 kHz
mono float32 requested directly (PipeWire resamples). Blocksize 1024
samples (64 ms) balances callback overhead against latency. If no audio
backend exists (headless), runa refuses with a clear message and offers
`--wav` offline mode.

## 6. Fallback policy

Build order: Vulkan → CPU (GGML_NATIVE/AVX2). The fallback is recorded
in `BUILD-MODE.txt` and surfaced in the README so Karl knows which
backend is live. Vulkan probe = `vulkaninfo --summary` must find an ICD
(RADV for the 780M / Barcelo).
