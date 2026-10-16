"""Command line interface."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import audio, clean as cleanup, confidence, report as health
from . import srt as srt_io
from . import tools, transcribe

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".ts", ".webm", ".flv", ".avi",
                  ".m4v", ".wmv", ".mpg", ".mpeg"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
MEDIA_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES

WORK_DIR = "_asmrsub"
# Windows refuses paths beyond 260 characters unless long paths are enabled,
# and media filenames are routinely 80+ characters long.
MAX_PATH = 255


def short_name(stem: str, limit: int = 40) -> str:
    """A stable, short stand-in for very long filenames."""
    if len(stem) <= limit:
        return stem
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:limit]}_{digest}"


@dataclass
class Outcome:
    name: str
    cues: int
    removed: int
    coverage: float
    seconds: float


def collect_media(targets: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in targets or ["."]:
        path = Path(raw).expanduser()
        if path.is_dir():
            found += sorted(p for p in path.iterdir()
                            if p.suffix.lower() in MEDIA_SUFFIXES)
        elif path.is_file():
            found.append(path)
        else:
            print(f"not found: {path}", file=sys.stderr)
    return found


def process(media: Path, chain: tools.Toolchain, args) -> Outcome | None:
    folder = media.resolve().parent
    stem = media.stem
    work = folder / WORK_DIR
    short = short_name(stem)

    final = folder / f"{stem}{args.suffix}.srt"
    wav = work / f"{short}.16k.wav"
    # whisper.cpp appends the format extension to whatever -of is given, so
    # build the result path by concatenation. Path.with_suffix would eat a
    # trailing ".raw" and point at the wrong file.
    raw_base = work / f"{short}_raw"
    raw_srt = Path(str(raw_base) + ".srt")

    longest = max(len(str(p)) for p in (final, wav, raw_base))
    if tools.WINDOWS and longest > MAX_PATH:
        print(f"  skipped: path would exceed {MAX_PATH} characters "
              f"({longest}). Move the file to a shorter directory.")
        return None

    if final.exists() and not args.force:
        print("  skipped: subtitles already exist (use --force to redo)")
        return None

    work.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not wav.exists() or args.force:
        print("  [1/3] extracting audio ...", end="", flush=True)
        if not audio.extract_for_asr(chain.ffmpeg, media, wav,
                                     work / f"{short}.ffmpeg.log"):
            print(" failed")
            return None
        print(" ok")
    else:
        print("  [1/3] reusing existing audio")

    if args.listen:
        mp3 = folder / f"{stem}.listening.mp3"
        if not mp3.exists() or args.force:
            print("  [+]   rendering listening mix ...", end="", flush=True)
            audio.extract_for_listening(chain.ffmpeg, media, mp3)
            print(" ok")

    options = transcribe.WhisperOptions(
        language=args.lang, use_vad=not args.no_vad,
        max_context=args.max_context, prompt=args.prompt,
        threads=args.threads, use_gpu=not args.no_gpu,
        gpu_device=args.device)

    if not raw_srt.exists() or args.force:
        label = "VAD on" if options.use_vad else "VAD OFF"
        label += ", CPU only" if args.no_gpu else ""
        label += f", {options.resolved_threads()} threads"
        print(f"  [2/3] transcribing ({label}) ...", end="", flush=True)
        ok = transcribe.run(chain.whisper, chain.model, wav, raw_base, options,
                            chain.vad_model, log=work / f"{short}.whisper.log")
        if not ok or not raw_srt.exists():
            print(f" failed (see {work / f'{short}.whisper.log'})")
            return None
        print(f" ok  {time.time() - started:.0f}s")
    else:
        print("  [2/3] reusing existing transcript")

    print("  [3/3] detecting silence and cleaning ...", end="", flush=True)
    silence = audio.detect_silence(chain.ffmpeg, wav)
    result = cleanup.clean_file(raw_srt, silence, final, args.repeat_limit)
    print(f" removed {result.removed_count}")
    for removal in result.removed[:6]:
        text = removal.cue.text.replace("\n", " ")[:34]
        print(f"        {srt_io.format_timestamp(removal.cue.start)}  "
              f"{removal.reason:<12} {text}")
    if result.removed_count > 6:
        print(f"        ... and {result.removed_count - 6} more")

    rep = health.analyse(result.kept, silence, audio.wav_duration(wav))
    elapsed = time.time() - started
    print(f"  done: {len(result.kept)} cues, speech coverage "
          f"{rep.coverage*100:.0f}%, {elapsed:.0f}s -> {final.name}")
    if rep.coverage < health.LOW_COVERAGE:
        print(f"  WARNING: coverage is only {rep.coverage*100:.0f}% - "
              "run `asmrsub report` for details")

    if args.remove_wav:
        wav.unlink(missing_ok=True)

    return Outcome(stem, len(result.kept), result.removed_count,
                   rep.coverage, elapsed)


def cmd_run(args) -> int:
    chain = tools.discover(args.ffmpeg, args.whisper, args.model,
                           args.vad_model, require_vad=not args.no_vad)
    print(chain.describe(), end="\n\n")

    media = collect_media(args.targets)
    if not media:
        print("no media files found")
        return 1

    print(f"{len(media)} file(s) to process\n" + "=" * 62)
    results: list[Outcome] = []
    for number, path in enumerate(media, 1):
        print(f"\n[{number}/{len(media)}] {path.name}")
        outcome = process(path, chain, args)
        if outcome:
            results.append(outcome)

    print("\n" + "=" * 62)
    if results:
        print(f"{'file':<34}{'cues':>7}{'removed':>9}{'cover':>8}{'time':>8}")
        for r in results:
            name = r.name if len(r.name) <= 33 else r.name[:32] + "…"
            print(f"{name:<34}{r.cues:>7}{r.removed:>9}"
                  f"{r.coverage*100:>7.0f}%{r.seconds:>7.0f}s")
    print(f"\ncompleted {len(results)}/{len(media)}")
    return 0 if results else 1


def cmd_clean(args) -> int:
    ffmpeg = tools.discover_ffmpeg_only(args.ffmpeg)
    silence = audio.detect_silence(ffmpeg, Path(args.media))
    destination = Path(args.output) if args.output else \
        Path(args.subtitles).with_suffix(".clean.srt")
    result = cleanup.clean_file(Path(args.subtitles), silence, destination,
                                args.repeat_limit)
    print(f"removed {result.removed_count}, kept {len(result.kept)}")
    for removal in result.removed:
        text = removal.cue.text.replace("\n", " ")[:40]
        print(f"  {srt_io.format_timestamp(removal.cue.start)}  "
              f"{removal.reason:<12} {text}")
    print(f"\nwrote {destination}")
    return 0


def cmd_report(args) -> int:
    ffmpeg = tools.discover_ffmpeg_only(args.ffmpeg)
    media = Path(args.media)
    silence = audio.detect_silence(ffmpeg, media)
    total = audio.duration(ffmpeg, media)
    print(health.analyse_file(Path(args.subtitles), silence, total).render())
    return 0


def cmd_confidence(args) -> int:
    ffmpeg = tools.discover_ffmpeg_only(args.ffmpeg)
    levels = audio.rms_per_second(ffmpeg, Path(args.media))
    result = confidence.analyse(Path(args.json), levels)
    if not result.segments:
        print("no token probabilities found - was the JSON produced with -ojf?")
        return 1
    print(result.render())
    print("\nleast confident segments:")
    for segment in sorted(result.segments, key=lambda s: s.confidence)[:12]:
        print(f"  {srt_io.format_timestamp(segment.start)}  "
              f"{segment.confidence:.2f}  {segment.rms:>6.1f}dB  "
              f"{segment.text[:36]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asmrsub",
        description="Batch subtitle generation tuned for quiet, long-form "
                    "speech, with VAD on by default and hallucination cleanup.")
    parser.add_argument("--ffmpeg", help="path to the ffmpeg binary")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="full pipeline over files or folders")
    run.add_argument("targets", nargs="*", help="media files or directories")
    run.add_argument("--lang", default="ja", help="spoken language (default: ja)")
    run.add_argument("--model", help="whisper model .bin")
    run.add_argument("--whisper", help="path to the whisper.cpp CLI")
    run.add_argument("--vad-model", help="path to ggml-silero-*.bin")
    run.add_argument("--no-vad", action="store_true",
                     help="disable VAD (not recommended - see README)")
    run.add_argument("--prompt", help="initial prompt; use it to pin down "
                                      "proper nouns and domain vocabulary")
    run.add_argument("--max-context", type=int, default=0,
                     help="whisper -mc; 0 prevents repetition loops")
    run.add_argument("--threads", type=int,
                     help="CPU threads; defaults to every logical core, "
                          "because whisper.cpp's own default of 4 leaves a "
                          "CPU-only machine mostly idle")
    run.add_argument("--no-gpu", action="store_true",
                     help="force CPU even if the whisper build supports GPU")
    run.add_argument("--device", type=int,
                     help="GPU device index for multi-GPU machines")
    run.add_argument("--repeat-limit", type=int,
                     default=cleanup.DEFAULT_REPEAT_LIMIT,
                     help="consecutive identical lines treated as a loop "
                          "(default 6; 0 disables)")
    run.add_argument("--suffix", default="",
                     help="appended to the output .srt stem")
    run.add_argument("--listen", action="store_true",
                     help="also render a dynamic-range-compressed MP3 for "
                          "human listening")
    run.add_argument("--remove-wav", action="store_true",
                     help="delete the intermediate 16 kHz WAV when done")
    run.add_argument("--force", action="store_true", help="redo finished work")
    run.set_defaults(func=cmd_run)

    clean = sub.add_parser("clean", help="strip invented cues from an SRT")
    clean.add_argument("subtitles")
    clean.add_argument("media", help="the audio the SRT was made from")
    clean.add_argument("-o", "--output")
    clean.add_argument("--repeat-limit", type=int,
                       default=cleanup.DEFAULT_REPEAT_LIMIT)
    clean.set_defaults(func=cmd_clean)

    rep = sub.add_parser("report", help="coverage and sanity check for an SRT")
    rep.add_argument("subtitles")
    rep.add_argument("media")
    rep.set_defaults(func=cmd_report)

    conf = sub.add_parser(
        "confidence",
        help="correlate token confidence with loudness (needs whisper -ojf)")
    conf.add_argument("json", help="whisper.cpp JSON from -ojf")
    conf.add_argument("media", help="the audio that was transcribed")
    conf.set_defaults(func=cmd_confidence)

    return parser


def main(argv: list[str] | None = None) -> int:
    tools.enable_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except tools.ToolNotFound as error:
        print(f"\n{error}\n", file=sys.stderr)
        print("See the Requirements section of the README for how to install "
              "these.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
