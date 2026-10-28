# Measurements

Findings 1–4 come from one 95-minute Japanese recording: long-form,
conversational, much of it spoken very quietly, with loud non-speech sounds
throughout. It is the kind of material that breaks the default settings of
most Whisper front-ends, which is why it was useful to measure.

Hardware: AMD RX 6650 XT (Vulkan), Intel i5-12400F, `large-v3-turbo`.

Findings 5 and 6 come from two further Japanese recordings — a 40-minute
whispered ASMR stream and a 154-minute recording with long stretches of
non-speech sound — on an Apple M2 (Metal), whisper.cpp 1.9.5,
`large-v3-turbo`, Silero VAD v6.2.0.

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

## Finding 5: VAD stretches cues across the gaps

Both recordings transcribed without errors and reported 91% and 98% coverage.
Both had lines sitting on screen for minutes.

| | 40-min stream | 154-min recording |
|---|---|---|
| cues | 64 | 492 |
| cues longer than 30 s | 16 | 61 |
| longest cue | 328.6 s | 484.2 s |
| VAD speech under that cue | 1.0 s | 1.6 s |
| cues carrying > 2 s of non-speech | 39 | 249 |

The mechanism is in how whisper.cpp runs VAD. It cuts out the speech segments,
joins them end to end with 0.1 s of silence between them, decodes that, and
maps each timestamp back to the original audio through a table of segment
boundaries. Inside each 0.1 s join the table interpolates linearly across the
*whole* original gap. A timestamp that lands in a join — routine, since that
is where whisper sees a pause — comes out somewhere in the middle of a gap
that may be minutes long. And a cue whose words straddle a join spans the gap
outright: whisper heard a 0.1 s pause, the subtitle covers 90 s of other
sound.

That makes the edges easy: any part of a cue that reaches past its speech into
a gap came from interpolation, and pulling it back is just undoing the map.
The whisper log lists every VAD segment, so `asmrsub` trims each cue that
reaches more than 2 s beyond its speech.

The judgement call is a cue whose speech sits in two islands far apart: which
one held the words? 69 cues looked like that. Each island was cut out and
re-transcribed on its own:

| | cues |
|---|---|
| words in one island only | 54 |
| — the island with the most speech | **48** |
| — always the last island would have been right | 46 |
| — always the first island would have been right | 6 |
| words split across two islands | 7 |
| inconclusive (island too short to transcribe alone) | 8 |

So a straddling cue keeps the island with the most speech. It is wrong about
one time in nine, and when it is, the line appears a few seconds or minutes
early or late — against being on screen for all of those minutes before.

Trimming also changed what coverage meant. Against non-silent audio, the
trimmed files scored 8% and 12%: nearly all of their non-silent audio was
not speech, and the stretched cues had been filling it. Measured against the
speech VAD found, they scored 74% and 88% (78% and 99% before trimming; the
difference is speech that only a stretched cue had covered). `asmrsub run` now
reports coverage that way whenever the log has the segments. Finding 1's VAD
figures were taken before trimming existed and may be flattered the same
way; by how much was not re-measured.

The same files had three decoder loops *inside* single cues — `あ、` repeated
77 to 109 times, one of them in a 316-second cue — which rule 3, looking for
runs of identical lines, cannot see. The most a genuine line repeated anything
was four times (`だんだんだんだん`, `ダメダメダメダメ`), so the line-level
threshold of 6 separates the two cleanly here as well.

## Finding 6: the default VAD threshold misses faint whispering

The stream's VAD found only 2.8 minutes of speech in 40. Lowering whisper.cpp's
speech threshold from its default of 0.5:

| `--vad-threshold` | VAD speech | cues | characters |
|---|---|---|---|
| 0.5 (default) | 2.8 min | 64 | 636 |
| 0.35 | 3.1 min | 85 | 848 |
| 0.2 | 3.8 min | 97 | 867 |

At 0.35 the transcript gained a third more text, including a 45-second passage
— viewers' names being read out — that the default run had no subtitles for at
all. Neither lower threshold produced a phantom line such as
`ご視聴ありがとうございました`. Individual lines came out differently
— some better, some worse — because the decoder sees different audio around
them. One recording is not enough to move the default, so it stays at 0.5 and
`--vad-threshold` is there for whispered material.

## Summary

Ranked by value actually delivered:

1. `--vad` — recovered 24.6 minutes of missing dialogue, and ran 3× faster
2. silence-based cleanup — removes invented lines deterministically
3. trimming cues to VAD speech — undoes the stretching VAD itself causes
4. `--prompt` — fixes proper nouns, free
5. SNR-preserving audio prep — correct, small effect
6. source separation — **measured at approximately zero benefit here**

The general lesson is in the gap between items 1 and 5. The intuitive
diagnosis (loud sounds drowning quiet speech) was wrong, and the actual defect
(whole windows being discarded) produced no error message and no visible
symptom. Measure coverage, and measure confidence against the variable you
suspect, before buying an expensive fix.
