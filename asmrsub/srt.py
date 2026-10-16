"""Minimal SRT reader/writer.

Deliberately forgiving: whisper.cpp, Subtitle Edit and hand edits all produce
slightly different files (BOM or not, CRLF or LF, ``.`` or ``,`` before the
milliseconds), and none of that is worth failing over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TIME_SPLIT = re.compile(r"\s*-->\s*")


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, _, millis = rest.replace(".", ",").partition(",")
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            + int(millis or 0) / 1000.0)


def format_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:            # rounding can carry into the next second
        millis, whole = 0, whole + 1
    return (f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:"
            f"{whole % 60:02d},{millis:03d}")


def read(path: str | Path) -> list[Cue]:
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    cues: list[Cue] = []
    for block in re.split(r"\n{2,}", raw.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        index = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if index is None:
            continue
        try:
            start_s, end_s = _TIME_SPLIT.split(lines[index], maxsplit=1)[:2]
            start = parse_timestamp(start_s.strip())
            # Some writers append positioning data after the end time.
            end = parse_timestamp(end_s.strip().split()[0])
        except (ValueError, IndexError):
            continue
        cues.append(Cue(start, end, "\n".join(lines[index + 1:]).strip()))
    return cues


def write(path: str | Path, cues: list[Cue]) -> None:
    out = []
    for number, cue in enumerate(cues, 1):
        out.append(f"{number}\n{format_timestamp(cue.start)} --> "
                   f"{format_timestamp(cue.end)}\n{cue.text}\n")
    # UTF-8 with BOM: the widest compatibility across desktop video players.
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8-sig")
