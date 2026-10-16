"""Locate the external binaries and models asmrsub depends on.

Resolution order for every item is the same:
  1. explicit argument (``--ffmpeg``, ``--whisper``, ``--model``, ``--vad-model``)
  2. environment variable (``ASMRSUB_FFMPEG`` etc.)
  3. ``PATH``
  4. a per-platform list of places these things are usually installed

Nothing here is Windows-specific by design; the Windows entries exist because
Subtitle Edit ships a ready-made whisper.cpp build plus the Silero VAD model,
which is the easiest way to get a working setup on that platform.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"

FFMPEG_NAMES = ("ffmpeg",)
FFPROBE_NAMES = ("ffprobe",)
# whisper.cpp renamed its CLI from `main` to `whisper-cli`; Homebrew ships it
# as `whisper-cpp`. Only the distinctive names are safe to look up on PATH -
# searching for a bare `main` matches things like Windows' main.CPL, because
# PATHEXT counts Control Panel applets as executable.
WHISPER_PATH_NAMES = ("whisper-cli", "whisper-cpp")
WHISPER_DIR_NAMES = ("whisper-cli", "whisper-cpp", "whisper", "main")

# Extensions that are genuinely runnable programs, as opposed to everything
# Windows is willing to "open" through PATHEXT.
RUNNABLE_SUFFIXES = {".exe", ".bat", ".cmd", ".com", ""}

VAD_GLOB = "ggml-silero*.bin"
MODEL_GLOB = "*.bin"


class ToolNotFound(RuntimeError):
    pass


def _subtitle_edit_roots() -> list[Path]:
    """Subtitle Edit bundles whisper.cpp binaries, models and the VAD model."""
    if not WINDOWS:
        return []
    roots = []
    for env in ("APPDATA", "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            roots.append(Path(base) / "Subtitle Edit")
    # Portable installs are common; probe a few conventional drive locations.
    for drive in "CDEF":
        roots += [Path(f"{drive}:/tool/Subtitle Edit"),
                  Path(f"{drive}:/Program Files/Subtitle Edit"),
                  Path(f"{drive}:/SubtitleEdit")]
    return roots


def _whisper_dirs() -> list[Path]:
    dirs: list[Path] = []
    # Prefer the Vulkan build: it is the one that works on AMD and Intel GPUs.
    for root in _subtitle_edit_roots():
        stt = root / "SpeechToText"
        dirs += [stt / "CppVulkan", stt / "Cpp", stt / "CppCuda"]
    home = Path.home()
    dirs += [home / "whisper.cpp" / "build" / "bin",
             home / "whisper.cpp",
             home / ".local" / "bin"]
    if MACOS:
        dirs += [Path("/opt/homebrew/bin"), Path("/usr/local/bin")]
    else:
        dirs += [Path("/usr/local/bin"), Path("/usr/bin")]
    return dirs


def _model_dirs() -> list[Path]:
    dirs = [Path.cwd() / "models"]
    for d in _whisper_dirs():
        dirs += [d, d / "Models", d / "models"]
    home = Path.home()
    dirs += [home / "whisper.cpp" / "models",
             home / ".cache" / "whisper.cpp",
             home / ".cache" / "whisper"]
    return dirs


def _exe(name: str) -> str:
    return name + ".exe" if WINDOWS else name


def _runnable(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in RUNNABLE_SUFFIXES


def _find_binary(path_names: Sequence[str], dirs: Iterable[Path],
                 dir_names: Sequence[str] | None = None) -> Path | None:
    """Look on PATH first, then in the directories we know about.

    ``path_names`` are distinctive enough to search PATH with; ``dir_names``
    may include generic names that are only safe inside a known install dir.
    """
    for name in path_names:
        found = shutil.which(name)
        if found and _runnable(Path(found)):
            return Path(found)
    for d in dirs:
        for name in (dir_names or path_names):
            cand = d / _exe(name)
            if _runnable(cand):
                return cand
    return None


def _find_model(dirs: Iterable[Path], pattern: str,
                exclude_vad: bool = False) -> Path | None:
    best: Path | None = None
    for d in dirs:
        if not d.is_dir():
            continue
        for cand in sorted(d.glob(pattern)):
            low = cand.name.lower()
            if exclude_vad and "silero" in low:
                continue
            if not exclude_vad and "silero" not in low:
                continue
            # Prefer large* models, otherwise keep the biggest file seen.
            if best is None:
                best = cand
            elif "large" in low and "large" not in best.name.lower():
                best = cand
            elif cand.stat().st_size > best.stat().st_size and \
                    ("large" in low) == ("large" in best.name.lower()):
                best = cand
    return best


def _resolve(explicit: str | None, env_var: str) -> Path | None:
    for value in (explicit, os.environ.get(env_var)):
        if value:
            p = Path(value).expanduser()
            if p.is_file():
                return p
            raise ToolNotFound(f"{env_var or 'path'} points at a missing file: {p}")
    return None


@dataclass
class Toolchain:
    ffmpeg: Path
    whisper: Path
    model: Path
    vad_model: Path | None

    def describe(self) -> str:
        vad = self.vad_model.name if self.vad_model else "(none - VAD disabled)"
        return (f"ffmpeg : {self.ffmpeg}\n"
                f"whisper: {self.whisper}\n"
                f"model  : {self.model.name}\n"
                f"vad    : {vad}")


def discover(ffmpeg: str | None = None, whisper: str | None = None,
             model: str | None = None, vad_model: str | None = None,
             require_vad: bool = True) -> Toolchain:
    ff = _resolve(ffmpeg, "ASMRSUB_FFMPEG") or _find_binary(FFMPEG_NAMES, [])
    wh = _resolve(whisper, "ASMRSUB_WHISPER") or _find_binary(
        WHISPER_PATH_NAMES, _whisper_dirs(), WHISPER_DIR_NAMES)
    md = _resolve(model, "ASMRSUB_MODEL") or _find_model(_model_dirs(), MODEL_GLOB,
                                                          exclude_vad=True)
    vd = _resolve(vad_model, "ASMRSUB_VAD_MODEL") or _find_model(_model_dirs(),
                                                                  VAD_GLOB)

    missing = []
    if ff is None:
        missing.append("ffmpeg (install it, or set ASMRSUB_FFMPEG)")
    if wh is None:
        missing.append("whisper.cpp CLI (set ASMRSUB_WHISPER)")
    if md is None:
        missing.append("a whisper model .bin (set ASMRSUB_MODEL)")
    if require_vad and vd is None:
        missing.append("the Silero VAD model ggml-silero-*.bin "
                       "(set ASMRSUB_VAD_MODEL, or pass --no-vad)")
    if missing:
        raise ToolNotFound("could not find:\n  - " + "\n  - ".join(missing))

    return Toolchain(ff, wh, md, vd)  # type: ignore[arg-type]


def discover_ffmpeg_only(ffmpeg: str | None = None) -> Path:
    """For the subcommands that only need ffmpeg, not a whisper setup."""
    found = _resolve(ffmpeg, "ASMRSUB_FFMPEG") or _find_binary(FFMPEG_NAMES, [])
    if found is None:
        raise ToolNotFound("could not find:\n  - ffmpeg "
                           "(install it, or set ASMRSUB_FFMPEG)")
    return found


def enable_utf8_console() -> None:
    """Make Chinese/Japanese output readable on a default Windows console."""
    if WINDOWS:
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # type: ignore[attr-defined]
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
