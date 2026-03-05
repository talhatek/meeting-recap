import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

from .audio import SUPPORTED_EXTENSIONS, discover_audio_files
from .summarizer import DEFAULT_SUMMARY_MODEL
from .transcriber import DEFAULT_MODEL, transcribe
from .summarizer import summarize


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="meeting-recap",
        description="Transcribe and summarize audio files using Groq API",
    )
    parser.add_argument(
        "-l", "--language",
        default="en",
        help="ISO-639-1 language code for the audio (default: en)",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model for transcription (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--summary-model",
        default=DEFAULT_SUMMARY_MODEL,
        help=f"LLM model for summarization (default: {DEFAULT_SUMMARY_MODEL})",
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

    # Load .env if present (no-op if missing)
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in your .env file or environment.")
        sys.exit(1)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    audio_files = discover_audio_files(input_dir)
    if not audio_files:
        print(f"No supported audio files found in '{input_dir}'.")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio file(s) in '{input_dir}'")
    print(f"Language       : {args.language}")
    print(f"Whisper model  : {args.model}")
    print(f"Summary model  : {args.summary_model}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_files:
        print(f"Processing: {audio_path.name}")
        print("-" * 60)

        # Transcribe
        print("  [1/2] Transcribing...")
        transcription = transcribe(
            audio_path,
            api_key=api_key,
            language=args.language,
            model=args.model,
            verbose=True,
        )

        if not transcription.strip():
            print("  WARNING: Empty transcription, skipping summarization.")
            print()
            continue

        stem = audio_path.stem
        transcription_path = output_dir / f"{stem}_transcription.txt"
        transcription_path.write_text(transcription.strip(), encoding="utf-8")
        print(f"  Transcription saved: {transcription_path}")

        # Summarize
        print("  [2/2] Summarizing...")
        summary = summarize(
            transcription,
            api_key=api_key,
            language=args.language,
            model=args.summary_model,
        )

        summary_path = output_dir / f"{stem}_summary.txt"
        summary_path.write_text(summary.strip(), encoding="utf-8")
        print(f"  Summary saved:       {summary_path}")
        print()

    print("Done! All files processed.")


if __name__ == "__main__":
    main()
