"""Meeting Recap -- Transcribe and summarize audio using Groq API.

Typical usage
-------------
Full pipeline (transcribe + summarize)::

    from meeting_recap import process

    result = process("meeting.mp3", api_key="gsk_...")
    print(result.transcription)
    print(result.summary)

Transcription only::

    from meeting_recap import transcribe

    text = transcribe("meeting.mp3", api_key="gsk_...", language="ru")

Summarization only::

    from meeting_recap import summarize

    summary = summarize(text, api_key="gsk_...", language="ru")
"""

from dataclasses import dataclass
from pathlib import Path

from .summarizer import DEFAULT_SUMMARY_MODEL, summarize
from .transcriber import DEFAULT_MODEL, transcribe

__all__ = ["transcribe", "summarize", "process", "RecapResult"]
__version__ = "0.1.0"


@dataclass
class RecapResult:
    """Holds the output of a full transcribe + summarize pipeline run."""

    audio_path: Path
    transcription: str
    summary: str


def process(
    audio_path: str | Path,
    api_key: str,
    language: str = "en",
    model: str = DEFAULT_MODEL,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    verbose: bool = True,
) -> RecapResult:
    """Run the full pipeline: transcribe an audio file and summarize it.

    Args:
        audio_path:    Path to the audio file.
        api_key:       Groq API key.
        language:      ISO-639-1 language code of the audio (e.g. ``"en"``, ``"ru"``).
        model:         Whisper model ID for transcription.
        summary_model: Groq chat model ID for summarization.
        verbose:       Print progress to stdout.

    Returns:
        A :class:`RecapResult` with ``transcription`` and ``summary`` fields.
    """
    audio_path = Path(audio_path)

    if verbose:
        print(f"[1/2] Transcribing {audio_path.name}...")
    transcription = transcribe(
        audio_path, api_key=api_key, language=language, model=model, verbose=verbose
    )

    if verbose:
        print("[2/2] Summarizing...")
    summary = summarize(
        transcription, api_key=api_key, language=language, model=summary_model
    )

    return RecapResult(audio_path=audio_path, transcription=transcription, summary=summary)
