"""Remove the subtitles a recogniser invents when nobody is speaking.

Three rules, in order of how much evidence backs them:

1. **Silent spans.** A cue that sits inside detected silence is fabricated,
   full stop. This is the rule that matters: on real material it catches the
   classic Japanese phantom line, zero-length cues, and the degenerate case of
   a single cue stretched across minutes of nothing.

2. **Zero-length cues.** ``00:12:15,000 --> 00:12:15,000`` is not a subtitle,
   and some players choke on it.

3. **Repetition runs.** A line repeated many times in a row is usually a
   decoder loop. The threshold defaults to 6 and should not be lowered:
   genuine speech repeats short interjections three to five times all the
   time, and a threshold of 3 deletes real content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .audio import Span
from .srt import Cue

DEFAULT_REPEAT_LIMIT = 6
DEFAULT_OVERLAP = 0.70

_WHITESPACE = re.compile(r"\s+")


@dataclass
class Removal:
    cue: Cue
    reason: str


@dataclass
class CleanResult:
    kept: list[Cue]
    removed: list[Removal]

    @property
    def removed_count(self) -> int:
        return len(self.removed)


def _silence_fraction(cue: Cue, silence: list[Span]) -> float:
    if cue.duration <= 0:
        return 1.0
    covered = sum(span.overlap(cue.start, cue.end) for span in silence)
    return covered / cue.duration


def clean(cues: list[Cue], silence: list[Span],
          repeat_limit: int = DEFAULT_REPEAT_LIMIT,
          overlap: float = DEFAULT_OVERLAP) -> CleanResult:
    kept: list[Cue] = []
    removed: list[Removal] = []

    run_text: str | None = None
    run_length = 0

    for cue in cues:
        reason: str | None = None
        if cue.duration <= 0:
            reason = "zero-length"
        elif _silence_fraction(cue, silence) >= overlap:
            reason = "in silence"

        normalised = _WHITESPACE.sub("", cue.text)
        if normalised and normalised == run_text:
            run_length += 1
        else:
            run_text, run_length = normalised, 1

        if reason is None and repeat_limit and run_length >= repeat_limit:
            reason = f"repeat x{run_length}"

        if reason:
            removed.append(Removal(cue, reason))
        else:
            kept.append(cue)

    return CleanResult(kept, removed)


def clean_file(srt_path: Path, silence: list[Span], destination: Path,
               repeat_limit: int = DEFAULT_REPEAT_LIMIT) -> CleanResult:
    from . import srt as srt_io

    result = clean(srt_io.read(srt_path), silence, repeat_limit)
    srt_io.write(destination, result.kept)
    return result
