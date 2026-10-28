# asmrsub

Batch subtitle generation for ASMR and other quiet, long-form recordings —
built around [whisper.cpp](https://github.com/ggml-org/whisper.cpp), with voice
activity detection on by default and automatic removal of the lines a
recogniser invents when nobody is speaking.

Whispered speech, hours-long sessions and loud non-speech sounds are exactly
the conditions under which a default Whisper setup fails *silently* — it
returns a subtitle file that looks correct while large stretches of dialogue
are simply absent from it. Everything here exists to catch that. Nothing is
ASMR-specific, so it works just as well on lectures, interviews or podcasts.

Point it at a folder of videos; get an `.srt` next to each one.

```console
$ asmrsub run ./recordings --lang ja
ffmpeg : /usr/bin/ffmpeg
whisper: /usr/local/bin/whisper-cli
model  : ggml-large-v3-turbo.bin
vad    : ggml-silero-v6.2.0.bin

1 file(s) to process
==============================================================

[1/1] lecture.mkv
  [1/3] extracting audio ... ok
  [2/3] transcribing (VAD on) ... ok  248s
  [3/3] detecting silence and cleaning ... removed 3
        00:11:54,960  in silence   ...
  done: 1025 cues, speech coverage 94%, 263s -> lecture.srt
```

## Why this exists

A plain whisper.cpp run on long, quietly-spoken audio can drop large stretches
of speech **without reporting anything wrong**. The output file looks fine:
sensible timings, plausible text, no errors. The speech is simply not in it.

Without VAD, whisper.cpp slices the input into fixed 30-second windows. A
window whose speech sits under louder non-speech gets scored as silence and
skipped wholesale. Measured on one 95-minute recording:

| | no VAD | `--vad` |
|---|---|---|
| speech with subtitles | 45% | **94%** |
| speech missed entirely | 24.6 min | **1.7 min** |
| invented lines in silence | 6 | 0 after cleanup |
| zero-length cues | 12 | 0 |
| wall-clock time | ~12 min | **4.1 min** |

VAD was also three times *faster*, because it skips the non-speech audio
instead of decoding it.

The second failure mode is the opposite one: during genuine silence a
recogniser will happily produce text anyway — in Japanese, famously
`ご視聴ありがとうございました` — sometimes as a single cue stretched across
minutes of nothing. `asmrsub` detects silent spans with ffmpeg and deletes
cues that fall inside them, which needs no model-side tuning and cannot be
fooled by a plausible-looking hallucination.

The third comes from VAD itself. whisper.cpp decodes the speech segments
joined end to end, then maps timestamps back to the original audio — and a cue
whose words straddle two segments ends up spanning everything between them.
On two further recordings, 77 cues stayed on screen for more than 30 seconds —
the longest held a line with 1.6 seconds of speech under it for 484 seconds —
and because every one of those seconds counted as "subtitled", coverage read a
reassuring 98%.
`asmrsub` reads the speech segments from whisper.cpp's log and trims each cue
back to its speech. See [docs/FINDINGS.md](docs/FINDINGS.md#finding-5-vad-stretches-cues-across-the-gaps).

## Install

```console
pip install git+https://github.com/Gre111/asmrsub
```

Or clone and run it in place — there are no Python dependencies beyond the
standard library:

```console
git clone https://github.com/Gre111/asmrsub
cd asmrsub
python -m asmrsub run ./recordings
```

## Requirements

Four external pieces. `asmrsub` finds them on `PATH` and in the usual install
locations; override any of them with a flag or an environment variable.

| What | Flag | Environment variable |
|---|---|---|
| ffmpeg | `--ffmpeg` | `ASMRSUB_FFMPEG` |
| whisper.cpp CLI | `--whisper` | `ASMRSUB_WHISPER` |
| whisper model `.bin` | `--model` | `ASMRSUB_MODEL` |
| Silero VAD model | `--vad-model` | `ASMRSUB_VAD_MODEL` |

**Linux / macOS**

```console
# macOS
brew install ffmpeg whisper-cpp

# Linux: ffmpeg from your package manager, whisper.cpp from source
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
cmake -B build && cmake --build build -j --config Release
sh ./models/download-ggml-model.sh large-v3-turbo
sh ./models/download-vad-model.sh silero-v6.2.0
```

**Windows**

The least painful route is [Subtitle Edit](https://www.nikse.dk/subtitleedit),
which downloads a ready-made whisper.cpp build, the models, *and* the Silero
VAD model for you. Run one transcription in its GUI, and `asmrsub` will find
everything afterwards. Prefer its **Vulkan** backend: it is the one that works
on AMD and Intel GPUs, where CUDA-based tools (faster-whisper and friends)
silently fall back to CPU.

## Hardware support

The compute backend is whatever your whisper.cpp binary was **built** with —
CUDA, Vulkan, Metal or plain CPU. `asmrsub` does not pick a backend and does
not need to: it runs the binary it finds, so a CUDA build uses CUDA, a Metal
build uses Metal, a CPU build uses the CPU. When several builds are installed
side by side it prefers CUDA, then Vulkan, then CPU, and prints the one it
chose on every run.

| | |
|---|---|
| NVIDIA | CUDA build, used automatically |
| AMD / Intel | Vulkan build — the portable GPU backend |
| Apple Silicon | Metal build (`brew install whisper-cpp` ships one) |
| No GPU | fully supported — see the threading note below |
| Multi-GPU | `--device N` |
| Force CPU | `--no-gpu` |

### CPU-only machines

whisper.cpp defaults `-t` to **4 threads** no matter how many cores are
present, which leaves most of a CPU-only machine idle. `asmrsub` defaults to
every logical core instead. Measured on a 6-core/12-thread desktop, 120 s of
audio, GPU disabled, `large-v3-turbo`:

| threads | time | speed |
|---|---|---|
| 4 (whisper.cpp's default) | 93.2 s | 1.3x realtime |
| 6 (physical cores) | 74.4 s | 1.6x |
| 8 | 63.6 s | 1.9x |
| **12 (all logical — the default here)** | **59.6 s** | **2.0x** |

Taking every logical thread won, so SMT is worth using despite the common
advice to stick to physical cores. Use `--threads` to leave headroom for other
work.

On a GPU build the setting makes no measurable difference — 9.9 s either way —
because the encoder runs on the GPU, so the raised default costs GPU users
nothing. Same clip, same machine:

| | time | speed |
|---|---|---|
| Vulkan, RX 6650 XT | 9.9 s | 12x realtime |
| best CPU result above | 59.6 s | 2.0x |

A full 95-minute recording took about 4 minutes on that GPU, roughly 23x
realtime.

CPU and GPU runs produce slightly different transcripts, because
floating-point arithmetic differs between backends. Expect small variation in
cue counts, not a difference in quality.

## Commands

### `asmrsub run`

The whole pipeline over files or folders. Already-finished work is skipped, so
re-running after adding new files is safe.

```console
asmrsub run ./recordings --lang ja
asmrsub run lecture.mkv --prompt "Kubernetes, etcd, kubelet"
asmrsub run ./in --listen --remove-wav
```

| Option | Meaning |
|---|---|
| `--lang` | spoken language (default `ja`) |
| `--prompt` | initial prompt; the fix for proper nouns the model keeps re-spelling |
| `--no-vad` | disable VAD — see the table above before you do |
| `--vad-threshold X` | VAD speech threshold (whisper.cpp's default 0.5); try 0.35 for faint whispering — [measured here](docs/FINDINGS.md#finding-6-the-default-vad-threshold-misses-faint-whispering) |
| `--repeat-limit N` | consecutive identical lines, or repeats inside one line, treated as a decoder loop (default 6) |
| `--no-trim` | keep whisper.cpp's cue timing even where a cue spans long non-speech |
| `--threads N` | CPU threads (default: every logical core) |
| `--no-gpu` / `--device N` | force CPU / pick a GPU |
| `--listen` | also render a compressed MP3 for human listening |
| `--remove-wav` | delete the intermediate 16 kHz WAV afterwards |
| `--force` | redo work that is already done |

### `asmrsub report`

Coverage and sanity check for a finished subtitle file. Coverage is measured
against *non-silent* audio, so a long legitimate pause does not look like a
failure.

```console
$ asmrsub report lecture.srt lecture.wav
cues            1025
subtitled       85.2 min of 90.5 min of speech  (94%)
total duration     95.5 min
detected silence    5.0 min
zero-length     0
inside silence  0
```

Anything under 60% triggers a warning; the first thing to check is whether VAD
was on.

Non-silent audio is only a stand-in for speech, and a poor one on material
full of non-speech sound — ASMR triggers, music, a massage. Pass the whisper
log from the run and coverage is measured against the speech VAD actually
found instead; `asmrsub run` always does this when the log has it.

```console
$ asmrsub report lecture.srt lecture.wav --whisper-log _asmrsub/lecture.whisper.log
```

The report also counts cues longer than 30 seconds. whisper.cpp decodes
30-second windows, so no segment it produces can be that long; a cue that is
was stretched across non-speech, and every second of it is counted as
"subtitled". Any such cue triggers a warning that coverage is overstated.

### `asmrsub clean`

Run the cleanup rules against a subtitle file you produced some other way —
Subtitle Edit, a web service, anything.

```console
asmrsub clean lecture.srt lecture.wav -o lecture.clean.srt
```

### `asmrsub confidence`

Answers one question: *is loud background noise actually hurting recognition?*
Needs a transcript produced with whisper.cpp's `-ojf` flag.

```console
asmrsub confidence transcript.json lecture.wav
```

It correlates per-token confidence with loudness. A clearly negative Pearson r
means masking is real and source separation may be worth its cost. Near zero or
positive means loudness is not the problem — see
[docs/FINDINGS.md](docs/FINDINGS.md) for a worked example where it came out at
**+0.161** and saved hours of pointless GPU time.

## How it works

**Audio preparation.** Mono sum, 16 kHz, 90 Hz highpass, mild stationary
denoise, then a **static** +20 dB gain and a limiter whose threshold sits well
above speech. The gain is static on purpose: it lifts speech and background by
the same amount, leaving the speech-to-noise ratio untouched. A dynamic
normaliser would raise quiet passages more than loud ones — better-looking
meters, worse recognition.

Binaural and stereo sources are summed rather than reduced to one channel,
because a voice that moves between channels would otherwise disappear whenever
it favours the discarded side.

**Transcription.** `--vad` with the Silero model, `--max-context 0` so the
decoder cannot feed its own output back in and fall into repetition loops, and
`--suppress-nst`.

**Cleanup.** Three removal rules, in order of how much evidence backs them:

1. cues sitting inside detected silence — fabricated, always
2. zero-length cues — not subtitles, and some players choke on them
3. runs of identical lines — a decoder loop, with the threshold at 6

That third threshold should not be lowered. Real speech repeats short
interjections three to five times constantly; a threshold of 3 deletes genuine
content. This was measured, not guessed.

Then two repairs to the cues that survive:

4. a loop *inside* one cue — `あ、` a hundred times over — is cut back to
   five repetitions, the same number rule 3 keeps of a run of lines. Genuine
   repetition within a line (`だんだんだんだん`) peaked at four in the
   measurements, so it is left alone.
5. a cue that reaches more than 2 s beyond the speech VAD found under it is
   trimmed back to that speech, plus a little padding and enough time to be
   read. When its speech sits in islands further apart than that, it keeps
   the island holding the most speech — re-transcribing each island on its
   own showed that to be where the words were 48 times out of 54. `--no-trim`
   turns this off.

## Output layout

```
recordings/
├── lecture.mkv
├── lecture.srt            <- next to the video, so players load it automatically
└── _asmrsub/             <- intermediates, safe to delete
    ├── lecture.16k.wav
    ├── lecture_raw.srt
    └── lecture.whisper.log
```

The raw transcript is kept so you can re-run cleanup with different settings
without paying for transcription again: delete the final `.srt` and run the
same command. The whisper log is kept for the same reason — it holds the VAD
speech segments that cue trimming works from.

## Limitations

- Timing is never stretched, shifted or re-synced, because subtitle timing has
  to match the original media. The one change is trimming cues back to the
  speech under them, inside the span whisper.cpp gave each cue.
- A cue whose words really do straddle two speech islands cannot be split
  without token-level timing, so it is shown with only one of them: the one
  holding most of its speech.
- Translation is out of scope. `asmrsub` produces subtitles in the spoken
  language.
- Very long filenames can exceed the Windows 260-character path limit.
  Intermediates use shortened names, and the file is skipped with a clear
  message rather than a traceback if the path is still too long.

## License

MIT
