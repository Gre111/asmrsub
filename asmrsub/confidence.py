"""Correlate per-token confidence with loudness.

Useful for one specific question: *is loud background noise actually hurting
recognition?* It is tempting to assume it must be, and to reach for source
separation to fix it. Measuring first is cheaper than being wrong.

On the recording this package was built against the answer was no: Pearson r
between segment loudness and mean token probability came out at **+0.161** -
slightly positive, meaning louder passages recognised marginally *better*. The
least confident segments were the quietest ones, and 72% of all low-confidence
segments were one- to eight-character interjections that no downstream
translation would care about. Source separation would have cost hours per file
to fix a problem that was not there.

Read the output like this:
  * r clearly negative  -> masking is real, separation may be worth it
  * r near zero or positive -> loudness is not the problem, look elsewhere
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

LOW_CONFIDENCE = 0.70

BUCKETS = (
    (-999.0, -45.0, "very quiet <-45dB"),
    (-45.0, -38.0, "-45..-38 dB"),
    (-38.0, -32.0, "-38..-32 dB"),
    (-32.0, -26.0, "-32..-26 dB"),
    (-26.0, -20.0, "-26..-20 dB"),
    (-20.0, 999.0, "loud >-20dB"),
)


@dataclass
class Segment:
    start: float
    end: float
    confidence: float
    rms: float
    text: str


@dataclass
class Bucket:
    label: str
    count: int
    confidence: float
    low_share: float


@dataclass
class ConfidenceReport:
    segments: list[Segment]
    buckets: list[Bucket] = field(default_factory=list)
    r: float = 0.0

    @property
    def mean_confidence(self) -> float:
        return statistics.fmean(s.confidence for s in self.segments)

    @property
    def low_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.confidence < LOW_CONFIDENCE]

    def render(self) -> str:
        lines = [f"segments analysed: {len(self.segments)}", ""]
        lines.append(f"{'loudness':<20}{'count':>7}{'mean conf':>12}{'low':>8}")
        for b in self.buckets:
            lines.append(f"{b.label:<20}{b.count:>7}{b.confidence:>12.3f}"
                         f"{b.low_share*100:>7.0f}%")

        low = self.low_segments
        lines += [
            "",
            f"Pearson r (loudness vs confidence): {self.r:+.3f}",
            "  negative -> louder passages recognise worse (masking is real)",
            "  ~zero    -> loudness is not what limits accuracy",
            "",
            f"mean confidence {self.mean_confidence:.3f}, "
            f"{len(low)/len(self.segments)*100:.0f}% of segments below "
            f"{LOW_CONFIDENCE}",
        ]

        short = sum(1 for s in low if len(s.text) < 9)
        if low:
            lines.append(f"of those low-confidence segments, "
                         f"{short/len(low)*100:.0f}% are under 9 characters "
                         f"(interjections, rarely worth fixing)")
        return "\n".join(lines)


def _mean_token_probability(segment: dict) -> float | None:
    tokens = [t for t in segment.get("tokens") or []
              if "p" in t and not str(t.get("text", "")).startswith("<|")]
    if not tokens:
        return None
    return statistics.fmean(float(t["p"]) for t in tokens)


def analyse(json_path: Path, rms_per_second: list[float | None]
            ) -> ConfidenceReport:
    """Parse whisper.cpp ``-ojf`` output and join it against loudness."""
    document = json.loads(Path(json_path).read_text(encoding="utf-8"))
    raw = document.get("transcription") or document.get("segments") or []

    segments: list[Segment] = []
    for entry in raw:
        offsets = entry.get("offsets") or {}
        start_ms, end_ms = offsets.get("from"), offsets.get("to")
        if start_ms is None or end_ms is None:
            continue
        confidence = _mean_token_probability(entry)
        if confidence is None:
            continue
        start, end = start_ms / 1000.0, end_ms / 1000.0
        window = [rms_per_second[i]
                  for i in range(int(start), min(int(end) + 1,
                                                 len(rms_per_second)))
                  if i < len(rms_per_second) and rms_per_second[i] is not None]
        if not window:
            continue
        segments.append(Segment(start, end, confidence,
                                statistics.fmean(window),  # type: ignore[arg-type]
                                (entry.get("text") or "").strip()))

    report = ConfidenceReport(segments)
    if not segments:
        return report

    for low, high, label in BUCKETS:
        group = [s for s in segments if low <= s.rms < high]
        if not group:
            continue
        report.buckets.append(Bucket(
            label, len(group),
            statistics.fmean(s.confidence for s in group),
            sum(1 for s in group if s.confidence < LOW_CONFIDENCE) / len(group)))

    xs = [s.rms for s in segments]
    ys = [s.confidence for s in segments]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs)
                            * sum((y - my) ** 2 for y in ys))
    report.r = numerator / denominator if denominator else 0.0
    return report
