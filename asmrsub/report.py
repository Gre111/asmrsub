"""Health check for a finished subtitle file.

The number worth watching is *coverage*: how much of the non-silent audio ended
up with subtitles. A recogniser that quietly skips a quarter of the recording
produces a file that looks perfectly fine — correct timings, plausible text,
no errors anywhere — and the only way to notice is to measure how much of the
audio it never accounted for.

Rough expectations: a talk-heavy recording should land well above 80%. Anything
under 60% means long stretches went missing, and the first thing to check is
whether VAD was enabled.

Coverage can also be overstated. whisper.cpp decodes 30-second windows, so no
segment it produces can be longer than that; a cue that is, was stretched
across non-speech when VAD timestamps were mapped back, and every second of it
counts as "subtitled". Those cues are counted separately so a high coverage
figure cannot hide them.

"Non-silent audio" is only a stand-in for speech. On material full of
non-speech sound - ASMR triggers, music, a massage - most of it is not speech,
and honest cue timing reads as very low coverage. When the speech segments
whisper.cpp's VAD found are known, coverage is measured against those instead.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path

from .audio import Span
from .srt import Cue, format_timestamp

LOW_COVERAGE = 0.60
GAP_SECONDS = 90.0
LONG_CUE_SECONDS = 30.0

_WHITESPACE = re.compile(r"\s+")


@dataclass
class Gap:
    start: float
    end: float
    silent: float
    speech: float | None = None   # VAD speech inside the gap, when known

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def unexplained(self) -> float:
        if self.speech is not None:
            return self.speech
        return self.length - self.silent


@dataclass
class Report:
    cue_count: int
    covered: float
    total: float
    silence_total: float
    zero_length: int
    in_silence: int
    long_cues: int = 0
    longest: float = 0.0
    speech_total: float | None = None     # VAD speech, when known
    speech_covered: float | None = None   # ... and how much of it has cues
    gaps: list[Gap] = field(default_factory=list)
    repeats: list[tuple[str, int]] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of the speech that has subtitles.

        Speech is what VAD found when that is known. Otherwise it is all
        *non-silent* audio: silence is excluded from the denominator on
        purpose, because a recording with a long legitimate pause is not
        missing anything, and measuring against raw duration would raise a
        false alarm on every such file.
        """
        if self.speech_total is not None:
            return (self.speech_covered or 0.0) / self.speech_total \
                if self.speech_total > 0 else 0.0
        speech = self.total - self.silence_total
        return self.covered / speech if speech > 0 else 0.0

    @property
    def raw_coverage(self) -> float:
        """Share of total duration, silence included."""
        return self.covered / self.total if self.total else 0.0

    @property
    def missing(self) -> float:
        return sum(gap.unexplained for gap in self.gaps)

    def render(self) -> str:
        if self.speech_total is not None:
            subtitled = (f"{(self.speech_covered or 0.0)/60:.1f} min of "
                         f"{self.speech_total/60:.1f} min of VAD speech")
        else:
            subtitled = (f"{self.covered/60:.1f} min of "
                         f"{(self.total-self.silence_total)/60:.1f} min of "
                         "speech")
        lines = [
            f"cues            {self.cue_count}",
            f"subtitled       {subtitled}  ({self.coverage*100:.0f}%)",
            f"total duration  {self.total/60:>7.1f} min",
            f"detected silence{self.silence_total/60:>7.1f} min",
            f"zero-length     {self.zero_length}",
            f"inside silence  {self.in_silence}",
            f"over {LONG_CUE_SECONDS:.0f}s        {self.long_cues}"
            f"  (longest {self.longest:.1f}s)",
        ]
        if self.gaps:
            lines.append("")
            lines.append(f"long gaps without subtitles (> {GAP_SECONDS:.0f}s):")
            for gap in self.gaps:
                lines.append(
                    f"  {format_timestamp(gap.start)} -> "
                    f"{format_timestamp(gap.end)}   {gap.length/60:5.2f} min, "
                    f"silent {gap.silent/60:.2f} min, "
                    f"unexplained {gap.unexplained/60:.2f} min")
            lines.append(f"  total unexplained: {self.missing/60:.1f} min")
        if self.repeats:
            lines.append("")
            lines.append("most repeated lines:")
            for text, count in self.repeats:
                lines.append(f"  {count:4d}x  {text[:46]}")
        if self.coverage < LOW_COVERAGE and self.speech_total is not None:
            lines.append("")
            lines.append(f"WARNING: only {self.coverage*100:.0f}% of the "
                         "speech VAD found has subtitles - whisper produced "
                         "nothing for the rest.")
        elif self.coverage < LOW_COVERAGE:
            lines.append("")
            lines.append(f"WARNING: coverage is only {self.coverage*100:.0f}%. "
                         "Large stretches of speech are probably missing - "
                         "check that VAD was enabled.")
        if self.long_cues:
            lines.append("")
            lines.append(f"WARNING: {self.long_cues} cue(s) last longer than "
                         f"{LONG_CUE_SECONDS:.0f}s, which no whisper segment "
                         "can - they were stretched across non-speech, and "
                         "coverage is overstated by them.")
        return "\n".join(lines)


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersection(a: list[tuple[float, float]],
                  b: list[tuple[float, float]]) -> float:
    """Total overlap of two sorted lists of disjoint spans, in one pass."""
    total, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyse(cues: list[Cue], silence: list[Span],
            total_seconds: float | None = None,
            speech: list[Span] | None = None) -> Report:
    total = total_seconds or max((c.end for c in cues), default=0.0)
    covered = sum(c.duration for c in cues if c.duration > 0)
    silence_total = sum(s.end - s.start for s in silence)

    speech_total = speech_covered = None
    if speech:
        spans = _union([(s.start, s.end) for s in speech])
        speech_total = sum(end - start for start, end in spans)
        speech_covered = _intersection(
            spans, _union([(c.start, c.end) for c in cues if c.duration > 0]))

    zero = sum(1 for c in cues if c.duration <= 0)
    inside = 0
    for cue in cues:
        if cue.duration <= 0:
            continue
        hit = sum(s.overlap(cue.start, cue.end) for s in silence)
        if hit / cue.duration >= 0.70:
            inside += 1

    gaps: list[Gap] = []
    ordered = sorted((c for c in cues if c.duration > 0), key=lambda c: c.start)
    for first, second in zip(ordered, ordered[1:]):
        if second.start - first.end > GAP_SECONDS:
            silent = sum(s.overlap(first.end, second.start) for s in silence)
            heard = None if not speech else \
                sum(s.overlap(first.end, second.start) for s in speech)
            gaps.append(Gap(first.end, second.start, silent, heard))

    counter = collections.Counter(_WHITESPACE.sub("", c.text) for c in cues
                                  if c.text.strip())
    repeats = [(t, n) for t, n in counter.most_common(8) if n >= 8]

    long_cues = sum(1 for c in cues if c.duration > LONG_CUE_SECONDS)
    longest = max((c.duration for c in cues), default=0.0)

    return Report(len(cues), covered, total, silence_total, zero, inside,
                  long_cues, longest, speech_total, speech_covered,
                  gaps, repeats)


def analyse_file(srt_path: Path, silence: list[Span],
                 total_seconds: float | None = None,
                 speech: list[Span] | None = None) -> Report:
    from . import srt as srt_io

    return analyse(srt_io.read(srt_path), silence, total_seconds, speech)
