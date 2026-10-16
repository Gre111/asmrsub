# Measurements

Everything here comes from one 95-minute Japanese recording: long-form,
conversational, much of it spoken very quietly, with loud non-speech sounds
throughout. It is the kind of material that breaks the default settings of
most Whisper front-ends, which is why it was useful to measure.

Hardware: AMD RX 6650 XT (Vulkan), Intel i5-12400F, `large-v3-turbo`.

## The recording

| | |
|---|---|
| duration | 95 min 29 s |
| integrated loudness | −37.8 LUFS |
| true peak | −0.8 dBTP |
| loudness range | 20.9 LU |
| digital silence | 5.0 min across 11 spans, one of them 3.75 min |

Integrated loudness of −37.8 LUFS is roughly 20 dB below a typical spoken-word
master, while the true peak was already near full scale. The peaks were short
transients, not sustained loud passages: the loudest single minute measured
−27 dB RMS against a −45 dB RMS typical speech level, a crest factor around
45 dB.

## Finding 1: without VAD, a quarter of the speech vanishes

The first transcription used sensible-looking settings and no VAD. It produced
1415 cues and looked entirely reasonable. It was missing 24.6 minutes of
dialogue.

| | no VAD | `--vad` |
|---|---|---|
| cues | 1415 | 1028 |
| speech with subtitles | 43.2 min (45%) | **85.2 min (94%)** |
| speech missed | 24.6 min | 1.7 min |
| cues inside silence | 6 | 3 (0 after cleanup) |
| zero-length cues | 12 | 0 |
| lines repeated ≥8× | 12 distinct | 1 distinct |
| wall clock | ~12 min | **4.1 min** |

The largest single gap ran from 62:57 to 78:54 — **15.9 minutes with no
subtitles at all and only 0.05 minutes of silence in it**. Re-running that
stretch alone with VAD produced 108 cues covering 90% of it, with coherent
dialogue from the first second to the last.

The mechanism: without VAD, whisper.cpp processes fixed 30-second windows. In
that stretch the non-speech sounds were the loudest in the entire recording —
15 dB above the speech — so each window scored as non-speech and was skipped.

Nothing in the output indicates this happened. The only way to catch it is to
measure how much of the audio ended up accounted for, which is what
`asmrsub report` does.

## Finding 2: audio preprocessing barely matters

Considerable effort went into preparing the audio before any of the above:
mono downmix, highpass, denoise, carefully SNR-preserving static gain. All of
it was correct, and none of it mattered much next to a single flag.

Two things are worth keeping anyway, because they are nearly free:

- **Static gain, not dynamic normalisation.** Measured on two sections 7.6 dB
  apart in level, a loudnorm-based chain compressed that gap to 4.6 dB — it
  lifted the quieter section 3 dB *more*, which is 3 dB of SNR thrown away.
  Static gain reproduced the original 7.6 dB spread exactly.
- **Sum stereo to mono, do not pick a channel.** Channel correlation was
  measured by comparing the mono sum against both channels: the sum landed at
  −51.7 dB where perfect correlation predicts −49.6 dB and full independence
  predicts −51.8 dB. The channels are effectively independent — two separate
  microphones — so there is no phase cancellation to fear, but the voice moves
  between them, so discarding either one loses speech.

## Finding 3: loudness was not limiting accuracy

The obvious next move after the above was source separation — isolate the
voice, drop the interfering sounds. On this material it would have cost a
2.5 GB dependency and 1.5–3 hours of CPU per file.

Measuring first was cheaper. whisper.cpp's `-ojf` output carries a probability
per token; joining those against per-second loudness gives a direct test of
whether louder passages recognise worse.

| loudness | segments | mean confidence | below 0.70 |
|---|---|---|---|
| very quiet < −45 dB | 102 | 0.740 | 31% |
| −45 … −38 dB | 201 | 0.756 | 28% |
| −38 … −32 dB | 306 | 0.808 | 15% |
| −32 … −26 dB | 318 | 0.807 | 16% |
| −26 … −20 dB | 89 | 0.811 | 18% |
| loud > −20 dB | 12 | 0.737 | 33% |

**Pearson r = +0.161.** Positive: louder segments recognised marginally
*better*. The masking hypothesis was wrong. The least confident segments were
the quietest ones — the problem was speech being faint, which source
separation does not fix.

Breaking down the 203 low-confidence segments by length:

| cue length | low-confidence | share of that length |
|---|---|---|
| 1–3 chars | 56 | 66% |
| 4–8 chars | 90 | 34% |
| 9–15 chars | 41 | 12% |
| 16+ chars | 16 | 5% |

72% of low-confidence segments were interjections of eight characters or
fewer. Only **16 substantial sentences** — 1.6% of the transcript — were both
long and uncertain.

## Finding 4: what those 16 sentences actually were

Reading them revealed a pattern that none of the statistics had surfaced: the
same proper noun transcribed three different ways across the file. Not an
audio problem at all — the model simply did not know the word.

The fix is `--prompt`, which costs nothing: list the proper nouns and domain
vocabulary and the model stops guessing. This is now the first thing to try
when a transcript is mostly good but keeps mangling specific terms.

## Summary

Ranked by value actually delivered:

1. `--vad` — recovered 24.6 minutes of missing dialogue, and ran 3× faster
2. silence-based cleanup — removes invented lines deterministically
3. `--prompt` — fixes proper nouns, free
4. SNR-preserving audio prep — correct, small effect
5. source separation — **measured at approximately zero benefit here**

The general lesson is in the gap between items 1 and 5. The intuitive
diagnosis (loud sounds drowning quiet speech) was wrong, and the actual defect
(whole windows being discarded) produced no error message and no visible
symptom. Measure coverage, and measure confidence against the variable you
suspect, before buying an expensive fix.
