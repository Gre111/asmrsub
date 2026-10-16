"""Health check for a finished subtitle file.

The number worth watching is *coverage*: how much of the non-silent audio ended
up with subtitles. A recogniser that quietly skips a quarter of the recording
produces a file that looks perfectly fine — correct timings, plausible text,
no errors anywhere — and the only way to notice is to measure how much of the
audio it never accounted for.

Rough expectations: a talk-heavy recording should land well above 80%. Anything
under 60% means long stretches went missing, and the first thing to check is
whether VAD was enabled.
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

_WHITESPACE = re.compile(r"\s+")


@dataclass
class Gap:
    start: float
    end: float
    silent: float

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def unexplained(self) -> float:
        return self.length - self.silent


@dataclass
class Report:
    cue_count: int
    covered: float
    total: float
    silence_total: float
    zero_length: int
    in_silence: int
    gaps: list[Gap] = field(default_factory=list)
    repeats: list[tuple[str, int]] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of *non-silent* audio that has subtitles.

        Silence is excluded from the denominator on purpose: a recording with
        a long legitimate pause is not missing anything, and measuring against
        raw duration would raise a false alarm on every such file.
        """
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
        lines = [
            f"cues            {self.cue_count}",
            f"subtitled       {self.covered/60:.1f} min of "
            f"{(self.total-self.silence_total)/60:.1f} min of speech"
            f"  ({self.coverage*100:.0f}%)",
            f"total duration  {self.total/60:>7.1f} min",
            f"detected silence{self.silence_total/60:>7.1f} min",
            f"zero-length     {self.zero_length}",
            f"inside silence  {self.in_silence}",
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
        if self.coverage < LOW_COVERAGE:
            lines.append("")
            lines.append(f"WARNING: coverage is only {self.coverage*100:.0f}%. "
                         "Large stretches of speech are probably missing - "
                         "check that VAD was enabled.")
        return "\n".join(lines)


def analyse(cues: list[Cue], silence: list[Span],
            total_seconds: float | None = None) -> Report:
    total = total_seconds or max((c.end for c in cues), default=0.0)
    covered = sum(c.duration for c in cues if c.duration > 0)
    silence_total = sum(s.end - s.start for s in silence)

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
            gaps.append(Gap(first.end, second.start, silent))

    counter = collections.Counter(_WHITESPACE.sub("", c.text) for c in cues
                                  if c.text.strip())
    repeats = [(t, n) for t, n in counter.most_common(8) if n >= 8]

    return Report(len(cues), covered, total, silence_total, zero, inside,
                  gaps, repeats)


def analyse_file(srt_path: Path, silence: list[Span],
                 total_seconds: float | None = None) -> Report:
    from . import srt as srt_io

    return analyse(srt_io.read(srt_path), silence, total_seconds)
