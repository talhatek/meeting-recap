import os
import sys
import argparse
import tempfile
import math
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB (free tier limit)
CHUNK_OVERLAP_MS = 5_000  # 5 seconds overlap between chunks


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def discover_audio_files(input_dir: Path) -> list[Path]:
    """Find all supported audio files in the input directory."""
    files = [
        f for f in sorted(input_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def estimate_chunk_duration(audio: AudioSegment, target_bytes: int) -> int:
    """Estimate how many milliseconds of audio fits within *target_bytes*.

    Uses the export bitrate (64 kbps mp3) to estimate chunk duration.
    """
    # 64 kbps = 8000 bytes/sec = 8 bytes/ms
    export_bytes_per_ms = 8
    chunk_duration_ms = int(target_bytes / export_bytes_per_ms)
    # Safety margin: use 90% of the estimate
    return int(chunk_duration_ms * 0.9)


def split_audio(audio_path: Path) -> list[Path]:
    """Split an audio file into chunks that each fit under MAX_FILE_SIZE_BYTES.

    Returns a list of temporary file paths. Caller is responsible for cleanup.
    """
    print("File exceeds 25 MB, splitting into chunks...")
    audio = AudioSegment.from_file(str(audio_path))
    total_duration_ms = len(audio)

    chunk_duration_ms = estimate_chunk_duration(audio, MAX_FILE_SIZE_BYTES)
    chunk_duration_ms = max(chunk_duration_ms, 30_000)  # at least 30 seconds

    step_ms = chunk_duration_ms - CHUNK_OVERLAP_MS
    num_chunks = math.ceil(total_duration_ms / step_ms)

    temp_dir = tempfile.mkdtemp(prefix="meeting_recap_")
    chunk_paths: list[Path] = []

    for i in range(num_chunks):
        start = i * step_ms
        end = min(start + chunk_duration_ms, total_duration_ms)
        chunk = audio[start:end]

        chunk_path = Path(temp_dir) / f"chunk_{i:03d}.mp3"
        chunk.export(str(chunk_path), format="mp3", parameters=["-ac", "1", "-ar", "16000", "-b:a", "64k"])

        # Verify exported chunk fits
        if chunk_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            print(f"WARNING: chunk {i} is {chunk_path.stat().st_size / 1024 / 1024:.1f} MB, retrying with smaller duration")
            chunk_path.unlink()
            half_dur = chunk_duration_ms // 2
            sub_step = half_dur - CHUNK_OVERLAP_MS
            sub_start = start
            sub_idx = 0
            while sub_start < end:
                sub_end = min(sub_start + half_dur, end)
                sub_chunk = audio[sub_start:sub_end]
                sub_path = Path(temp_dir) / f"chunk_{i:03d}_{sub_idx:02d}.mp3"
                sub_chunk.export(str(sub_path), format="mp3", parameters=["-ac", "1", "-ar", "16000", "-b:a", "64k"])
                chunk_paths.append(sub_path)
                sub_start += sub_step
                sub_idx += 1
        else:
            chunk_paths.append(chunk_path)

        if end >= total_duration_ms:
            break

    print(f"Split into {len(chunk_paths)} chunks")
    return chunk_paths


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_file(client: Groq, file_path: Path, language: str, model: str) -> str:
    """Transcribe a single audio file using Groq Whisper API."""
    with open(file_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(file_path.name, f.read()),
            model=model,
            language=language,
            response_format="text",
            temperature=0.0,
        )
    return transcription


def transcribe_audio(client: Groq, audio_path: Path, language: str, model: str) -> str:
    """Transcribe an audio file, chunking if necessary."""
    file_size = audio_path.stat().st_size
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")

    if file_size <= MAX_FILE_SIZE_BYTES:
        print("Transcribing directly...")
        return transcribe_file(client, audio_path, language, model)

    # Need to chunk
    chunk_paths = split_audio(audio_path)
    try:
        transcriptions = []
        for i, chunk_path in enumerate(chunk_paths):
            print(f"Transcribing chunk {i + 1}/{len(chunk_paths)}...")
            text = transcribe_file(client, chunk_path, language, model)
            transcriptions.append(text.strip())

        full_text = " ".join(transcriptions)
        return full_text
    finally:
        # Clean up temp files
        for p in chunk_paths:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            chunk_paths[0].parent.rmdir()
        except (OSError, IndexError):
            pass


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize_text(client: Groq, text: str, language: str, summary_model: str) -> str:
    """Summarize transcribed text using Groq LLM."""
    lang_names = {
        "ru": "Russian", "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "it": "Italian", "pt": "Portuguese", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "tr": "Turkish",
        "uk": "Ukrainian", "pl": "Polish", "nl": "Dutch", "sv": "Swedish",
    }
    lang_name = lang_names.get(language, language)

    system_prompt = (
        f"You are a professional meeting summarizer. "
        f"The following is a transcription of a meeting/recording in {lang_name}. "
        f"Produce a clear, well-structured summary in {lang_name} that includes:\n"
        f"1. A brief overview of the meeting/recording\n"
        f"2. Key discussion points\n"
        f"3. Decisions made\n"
        f"4. Action items (if any)\n"
        f"5. Important details or deadlines mentioned\n\n"
        f"Keep the summary concise but comprehensive. Write in {lang_name}."
    )

    response = client.chat.completions.create(
        model=summary_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the transcription to summarize:\n\n{text}"},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Meeting Recap: Transcribe and summarize audio files using Groq API"
    )
    parser.add_argument(
        "-l", "--language",
        default="en",
        help="ISO-639-1 language code for the audio (default: en)",
    )
    parser.add_argument(
        "-m", "--model",
        default="whisper-large-v3-turbo",
        help="Whisper model to use for transcription (default: whisper-large-v3-turbo)",
    )
    parser.add_argument(
        "--summary-model",
        default="llama-3.3-70b-versatile",
        help="LLM model for summarization (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Directory containing audio files (default: input)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write results (default: output)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env file or environment variables.")
        sys.exit(1)

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Resolve paths
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Discover audio files
    audio_files = discover_audio_files(input_dir)
    if not audio_files:
        print(f"No supported audio files found in '{input_dir}'.")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio file(s) in '{input_dir}'")
    print(f"Language: {args.language}")
    print(f"Transcription model: {args.model}")
    print(f"Summary model: {args.summary_model}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_files:
        print(f"Processing: {audio_path.name}")
        print("-" * 60)

        # Step 1: Transcribe
        print("[1/2] Transcribing...")
        transcription = transcribe_audio(client, audio_path, args.language, args.model)

        if not transcription or not transcription.strip():
            print("WARNING: Empty transcription, skipping summarization.")
            print()
            continue

        # Write transcription
        stem = audio_path.stem
        transcription_path = output_dir / f"{stem}_transcription.txt"
        transcription_path.write_text(transcription.strip(), encoding="utf-8")
        print(f"Transcription saved: {transcription_path}")

        # Step 2: Summarize
        print("[2/2] Summarizing...")
        summary = summarize_text(client, transcription, args.language, args.summary_model)

        # Write summary
        summary_path = output_dir / f"{stem}_summary.txt"
        summary_path.write_text(summary.strip(), encoding="utf-8")
        print(f"Summary saved: {summary_path}")
        print()

    print("Done! All files processed.")


if __name__ == "__main__":
    main()
