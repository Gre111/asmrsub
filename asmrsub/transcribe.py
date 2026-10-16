"""Drive the whisper.cpp CLI.

The defaults here are the whole point of this package. On long-form audio with
quiet speech, a plain whisper.cpp run silently drops large stretches: without
VAD it slices the input into fixed 30-second windows, and a window whose speech
is buried under louder non-speech gets classified as silence and skipped
wholesale. Measured on a 95-minute recording, that cost 24.6 minutes of real
dialogue. Turning VAD on brought the loss down to 1.7 minutes.

``--max-context 0`` stops the decoder from feeding its own previous output back
in as a prompt, which is what makes it fall into "repeat the same line forever"
loops. ``--suppress-nst`` suppresses non-speech tokens.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def default_threads() -> int:
    """whisper.cpp defaults ``-t`` to 4 regardless of how many cores exist,
    which leaves most of a CPU-only machine idle.

    Measured on a 6-core/12-thread desktop, 120 s of audio, GPU disabled:

        4 threads  93.2 s    6 threads  74.4 s
        8 threads  63.6 s   12 threads  59.6 s

    Every logical thread won, so SMT is worth taking here even though the
    common advice is to use physical cores. On a GPU build the setting makes
    no measurable difference (9.9 s either way) because the encoder runs on
    the GPU, so raising the default costs GPU users nothing.
    """
    return os.cpu_count() or 4


@dataclass
class WhisperOptions:
    language: str = "ja"
    use_vad: bool = True
    max_context: int = 0
    suppress_non_speech: bool = True
    prompt: str | None = None
    threads: int | None = None
    use_gpu: bool = True
    gpu_device: int | None = None
    extra: list[str] = field(default_factory=list)

    def resolved_threads(self) -> int:
        return self.threads if self.threads else default_threads()


def build_command(whisper: Path, model: Path, audio: Path, out_base: Path,
                  options: WhisperOptions, vad_model: Path | None,
                  json_output: bool = False) -> list[str]:
    cmd = [str(whisper), "-m", str(model), "-f", str(audio),
           "-l", options.language,
           "-mc", str(options.max_context),
           "-of", str(out_base)]

    cmd.append("-ojf" if json_output else "-osrt")

    if options.use_vad:
        if vad_model is None:
            raise ValueError("VAD requested but no VAD model was found")
        cmd += ["--vad", "-vm", str(vad_model)]
    if options.suppress_non_speech:
        cmd.append("-sns")
    if options.prompt:
        cmd += ["--prompt", options.prompt]

    cmd += ["-t", str(options.resolved_threads())]
    if not options.use_gpu:
        cmd.append("-ng")
    elif options.gpu_device is not None:
        cmd += ["-dev", str(options.gpu_device)]

    cmd += options.extra
    return cmd


def run(whisper: Path, model: Path, audio: Path, out_base: Path,
        options: WhisperOptions, vad_model: Path | None,
        json_output: bool = False, log: Path | None = None) -> bool:
    cmd = build_command(whisper, model, audio, out_base, options, vad_model,
                        json_output)
    if log is not None:
        with open(log, "wb") as handle:
            code = subprocess.run(cmd, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
    else:
        code = subprocess.run(cmd).returncode
    return code == 0
