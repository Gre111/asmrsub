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

Two repairs then apply to the cues that survive:

4. **Loops inside a cue.** The same decoder loop also happens within a single
   cue - ``あ、`` a hundred times over. Any stretch repeated ``repeat_limit``
   times or more in a row is cut back to ``repeat_limit - 1`` repetitions,
   which is exactly what rule 3 keeps of a run of lines. Genuine repetition
   inside a line (``だんだんだんだん``) stays well under the limit.

5. **Cues stretched across non-speech.** With VAD on, whisper.cpp decodes the
   speech segments joined end to end with 0.1 s of silence between them, then
   maps every timestamp back to the original audio. A timestamp that lands in
   one of those 0.1 s joins is interpolated across the *whole* original gap,
   which may be minutes long, and a cue whose words straddle a join spans the
   gap outright: one short line ends up on screen for minutes. The VAD
   segments whisper.cpp logs say where the speech actually is, so a cue that
   reaches more than ``MAX_NON_SPEECH`` beyond its speech is pulled back to it,
   and a cue whose speech sits in islands further apart than that keeps the
   island holding most of it. Cues only ever shrink; nothing moves outside the
   span whisper gave it.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path

from .audio import Span
from .srt import Cue

DEFAULT_REPEAT_LIMIT = 6
DEFAULT_OVERLAP = 0.70

# Rule 5. A pause inside a spoken line rarely reaches two seconds, and with
# VAD on whisper never hears one that long anyway: it is compressed to 0.1 s.
MAX_NON_SPEECH = 2.0
PAD_BEFORE = 0.2
PAD_AFTER = 0.5
MIN_DISPLAY = 1.0
SECONDS_PER_CHAR = 0.15

_WHITESPACE = re.compile(r"\s+")


@dataclass
class Removal:
    cue: Cue
    reason: str


@dataclass
class Repair:
    before: Cue
    after: Cue
    reason: str


@dataclass
class CleanResult:
    kept: list[Cue]
    removed: list[Removal]
    repaired: list[Repair]

    @property
    def removed_count(self) -> int:
        return len(self.removed)

    def repaired_count(self, reason: str) -> int:
        return sum(1 for r in self.repaired if r.reason.startswith(reason))


def _silence_fraction(cue: Cue, silence: list[Span]) -> float:
    if cue.duration <= 0:
        return 1.0
    covered = sum(span.overlap(cue.start, cue.end) for span in silence)
    return covered / cue.duration


def _loop_pattern(repeat_limit: int) -> re.Pattern | None:
    if repeat_limit < 2:
        return None
    return re.compile(r"(.{1,30}?)\1{%d,}" % (repeat_limit - 1), re.DOTALL)


def _collapse_loops(text: str, pattern: re.Pattern,
                    repeat_limit: int) -> tuple[str, int]:
    """Rule 4. Returns the new text and the longest repeat count cut back."""
    longest = 0

    def shorten(match: re.Match) -> str:
        nonlocal longest
        unit = match.group(1)
        longest = max(longest, len(match.group(0)) // len(unit))
        return unit * (repeat_limit - 1)

    return pattern.sub(shorten, text), longest


def _speech_islands(cue: Cue, speech: list[Span],
                    starts: list[float]) -> list[list[float]]:
    """The VAD speech inside ``cue`` as ``[start, end, seconds of speech]``,
    merged across pauses no longer than MAX_NON_SPEECH."""
    islands: list[list[float]] = []
    first = max(bisect.bisect_right(starts, cue.start) - 1, 0)
    for span in speech[first:]:
        if span.start >= cue.end:
            break
        start, end = max(span.start, cue.start), min(span.end, cue.end)
        if end <= start:
            continue
        if islands and start - islands[-1][1] <= MAX_NON_SPEECH:
            islands[-1][1] = end
            islands[-1][2] += end - start
        else:
            islands.append([start, end, end - start])
    return islands


def _trim(cue: Cue, speech: list[Span], starts: list[float]) -> Cue | None:
    """Rule 5. The cue pulled back to its speech, or None to leave it be."""
    islands = _speech_islands(cue, speech, starts)
    if not islands:
        return None                       # no evidence either way
    if (len(islands) == 1
            and islands[0][0] - cue.start <= MAX_NON_SPEECH
            and cue.end - islands[0][1] <= MAX_NON_SPEECH):
        return None

    # Words that straddle a join cannot be split without token timing, so the
    # cue goes where most of its speech is. Checked by re-transcribing each
    # island of 54 such cues on its own: that island held the words 48 times,
    # against 46 for always taking the last island and 6 for the first.
    island = max(islands, key=lambda i: i[2])
    start = max(cue.start, island[0] - PAD_BEFORE)
    end = min(cue.end, island[1] + PAD_AFTER)

    # Short speech still needs to stay up long enough to be read; take the
    # time from the stretch whisper gave the cue, after first and then before.
    wanted = max(MIN_DISPLAY,
                 SECONDS_PER_CHAR * len(_WHITESPACE.sub("", cue.text)))
    if end - start < wanted:
        end = min(cue.end, start + wanted)
        start = max(cue.start, end - wanted)
    if end - start >= cue.duration:
        return None
    return Cue(start, end, cue.text)


def clean(cues: list[Cue], silence: list[Span],
          repeat_limit: int = DEFAULT_REPEAT_LIMIT,
          overlap: float = DEFAULT_OVERLAP,
          speech: list[Span] | None = None) -> CleanResult:
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

    # The repairs run on survivors only, so what gets removed is decided on
    # whisper's original timing exactly as before.
    repaired: list[Repair] = []
    loops = _loop_pattern(repeat_limit)
    speech = sorted(speech or [], key=lambda s: s.start)
    starts = [s.start for s in speech]
    for index, cue in enumerate(kept):
        if loops is not None:
            text, count = _collapse_loops(cue.text, loops, repeat_limit)
            if count:
                repaired.append(Repair(cue, Cue(cue.start, cue.end, text),
                                       f"loop x{count}"))
                cue = repaired[-1].after
        if speech:
            trimmed = _trim(cue, speech, starts)
            if trimmed is not None:
                repaired.append(Repair(cue, trimmed, "trimmed"))
                cue = trimmed
        kept[index] = cue

    return CleanResult(kept, removed, repaired)


def clean_file(srt_path: Path, silence: list[Span], destination: Path,
               repeat_limit: int = DEFAULT_REPEAT_LIMIT,
               speech: list[Span] | None = None) -> CleanResult:
    from . import srt as srt_io

    result = clean(srt_io.read(srt_path), silence, repeat_limit,
                   speech=speech)
    srt_io.write(destination, result.kept)
    return result
