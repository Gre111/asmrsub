"""Audio extraction and silence detection via ffmpeg.

Two filter chains, for two different jobs:

``ASR_CHAIN`` prepares audio for speech recognition. The guiding rule is that
nothing may change the speech-to-noise ratio, because that ratio is what
recognition accuracy actually depends on. So the gain is *static*: it lifts
speech and background by exactly the same amount. A dynamic normaliser would
lift quiet passages more than loud ones, which flatters a level meter and
quietly destroys SNR.

``LISTEN_CHAIN`` is the opposite: it is for human ears, and compresses the
dynamic range hard so whispered passages are audible without the loud moments
becoming painful. Never feed its output to a recogniser.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# mono sum -> 16 kHz (whisper's native rate, and an implicit 8 kHz lowpass)
# -> 90 Hz highpass (rumble) -> mild stationary denoise -> static +20 dB
# -> limiter whose threshold sits ~21 dB above speech, so it only catches
#    percussive transients and never touches the voice.
ASR_CHAIN = (
    "pan=mono|c0=.5*c0+.5*c1,"
    "aresample=16000,"
    "highpass=f=90,"
    "afftdn=nr=10:nf=-45:tn=1,"
    "volume=20dB,"
    "alimiter=limit=0.891:attack=5:release=50:level=disabled"
)

LISTEN_CHAIN = (
    "highpass=f=40,"
    "acompressor=threshold=0.0126:ratio=4:attack=10:release=250:knee=6"
    ":detection=rms:link=average,"
    "acompressor=threshold=0.1:ratio=8:attack=2:release=100:knee=4"
    ":detection=peak:link=maximum,"
    "loudnorm=I=-17:TP=-1.5:LRA=7,"
    "alimiter=limit=0.94:attack=5:release=50:level=disabled"
)

SILENCE_DB = -50.0
SILENCE_MIN_SECONDS = 3.0

_SILENCE_START = re.compile(r"silence_start: (-?[0-9.]+)")
_SILENCE_END = re.compile(r"silence_end: (-?[0-9.]+)")


@dataclass(frozen=True)
class Span:
    start: float
    end: float

    def overlap(self, start: float, end: float) -> float:
        return max(0.0, min(end, self.end) - max(start, self.start))


def _run(cmd: list[str], log: Path | None = None) -> int:
    if log is not None:
        with open(log, "wb") as handle:
            return subprocess.run(cmd, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode


def extract_for_asr(ffmpeg: Path, source: Path, destination: Path,
                    log: Path | None = None) -> bool:
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-i", str(source), "-vn", "-af", ASR_CHAIN,
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(destination)]
    return _run(cmd, log) == 0


def extract_for_listening(ffmpeg: Path, source: Path, destination: Path,
                          bitrate: str = "320k") -> bool:
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-i", str(source), "-vn", "-af", LISTEN_CHAIN,
           "-c:a", "libmp3lame", "-b:a", bitrate, "-ar", "48000",
           str(destination)]
    return _run(cmd, None) == 0


def detect_silence(ffmpeg: Path, media: Path, threshold_db: float = SILENCE_DB,
                   min_seconds: float = SILENCE_MIN_SECONDS) -> list[Span]:
    """Return the spans ffmpeg considers silent.

    These spans are where recognisers invent text, so they drive the cleanup
    step. Detection runs on the prepared audio, not the original, so the
    threshold is relative to the levels the recogniser actually saw.
    """
    cmd = [str(ffmpeg), "-hide_banner", "-nostdin", "-i", str(media),
           "-af", f"silencedetect=noise={threshold_db}dB:d={min_seconds}",
           "-f", "null", "-"]
    stderr = subprocess.run(cmd, capture_output=True, text=True,
                            errors="replace").stderr
    starts = [float(x) for x in _SILENCE_START.findall(stderr)]
    ends = [float(x) for x in _SILENCE_END.findall(stderr)]
    return [Span(s, e) for s, e in zip(starts, ends)]


def wav_duration(path: Path) -> float:
    """Exact length of a PCM WAV, read from its header - no decoding pass."""
    import wave

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        return handle.getnframes() / rate if rate else 0.0


def duration(ffmpeg: Path, media: Path) -> float:
    """Media length in seconds, read back through ffmpeg to avoid needing
    ffprobe as a second hard dependency."""
    cmd = [str(ffmpeg), "-hide_banner", "-nostdin", "-i", str(media),
           "-f", "null", "-"]
    stderr = subprocess.run(cmd, capture_output=True, text=True,
                            errors="replace").stderr
    match = None
    for match in re.finditer(r"time=(\d+):(\d\d):(\d\d\.\d+)", stderr):
        pass                      # the final progress line holds the total
    if not match:
        return 0.0
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def rms_per_second(ffmpeg: Path, media: Path) -> list[float | None]:
    """Mean RMS level in dB for every second of audio.

    ``None`` marks digital silence, where RMS is -inf. Used by the confidence
    report to test whether loud passages actually recognise worse.
    """
    cmd = [str(ffmpeg), "-hide_banner", "-nostdin", "-i", str(media), "-af",
           "asetnsamples=n=16000,astats=metadata=1:reset=1,"
           "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
           "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    values: list[float | None] = []
    for raw in re.findall(r"RMS_level=(-?[0-9.]+|-?inf)", result.stdout):
        values.append(None if "inf" in raw else float(raw))
    return values
