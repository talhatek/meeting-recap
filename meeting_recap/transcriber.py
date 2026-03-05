from pathlib import Path

from groq import Groq

from .audio import (
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    cleanup_chunks,
    split_audio,
)

DEFAULT_MODEL = "whisper-large-v3-turbo"


def transcribe(
    audio_path: str | Path,
    api_key: str,
    language: str = "en",
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> str:
    """Transcribe an audio file using Groq Whisper API.

    Automatically splits files larger than 25 MB into overlapping chunks
    and merges the results.

    Args:
        audio_path: Path to the audio file.
        api_key:    Groq API key.
        language:   ISO-639-1 language code of the audio (e.g. ``"en"``, ``"ru"``).
        model:      Whisper model ID to use.
        verbose:    Print progress to stdout.

    Returns:
        Full transcription as a plain string.
    """
    client = Groq(api_key=api_key)
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{audio_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_size = audio_path.stat().st_size
    if verbose:
        print(f"  File size: {file_size / 1024 / 1024:.1f} MB")

    if file_size <= MAX_FILE_SIZE_BYTES:
        if verbose:
            print("  Transcribing directly...")
        return _transcribe_single(client, audio_path, language, model)

    # Large file: chunk and merge
    chunk_paths = split_audio(audio_path, verbose=verbose)
    try:
        parts: list[str] = []
        for i, chunk_path in enumerate(chunk_paths):
            if verbose:
                print(f"  Transcribing chunk {i + 1}/{len(chunk_paths)}...")
            parts.append(_transcribe_single(client, chunk_path, language, model).strip())
        return " ".join(parts)
    finally:
        cleanup_chunks(chunk_paths)


def _transcribe_single(
    client: Groq, file_path: Path, language: str, model: str
) -> str:
    """Send a single file to the Groq transcription endpoint."""
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(file_path.name, f.read()),
            model=model,
            language=language,
            response_format="text",
            temperature=0.0,
        )
    # result is a str when response_format="text"
    return str(result)
