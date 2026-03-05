import math
import tempfile
from pathlib import Path

from pydub import AudioSegment

SUPPORTED_EXTENSIONS = {
    ".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"
}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB (free tier limit)
CHUNK_OVERLAP_MS = 5_000  # 5 seconds overlap between chunks


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Return all supported audio files in *input_dir*, sorted by name."""
    return [
        f for f in sorted(input_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _estimate_chunk_duration_ms(target_bytes: int) -> int:
    """Estimate chunk duration in ms that fits within *target_bytes* at 64 kbps.

    64 kbps = 8 000 bytes/sec = 8 bytes/ms. Apply a 10 % safety margin.
    """
    return int((target_bytes / 8) * 0.9)


def split_audio(audio_path: Path, verbose: bool = True) -> list[Path]:
    """Split *audio_path* into overlapping chunks under MAX_FILE_SIZE_BYTES.

    Each chunk is exported as 64 kbps mono MP3 at 16 kHz (optimal for Whisper).
    Returns a list of temporary file paths -- caller must clean them up.
    """
    if verbose:
        print("  File exceeds 25 MB, splitting into chunks...")

    audio = AudioSegment.from_file(str(audio_path))
    total_ms = len(audio)

    chunk_ms = max(_estimate_chunk_duration_ms(MAX_FILE_SIZE_BYTES), 30_000)
    step_ms = chunk_ms - CHUNK_OVERLAP_MS

    temp_dir = Path(tempfile.mkdtemp(prefix="meeting_recap_"))
    chunk_paths: list[Path] = []

    i = 0
    while True:
        start = i * step_ms
        end = min(start + chunk_ms, total_ms)
        chunk = audio[start:end]

        chunk_path = temp_dir / f"chunk_{i:03d}.mp3"
        _export_chunk(chunk, chunk_path)

        if chunk_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            # Fallback: halve the duration and re-split this window
            if verbose:
                print(
                    f"  WARNING: chunk {i} is "
                    f"{chunk_path.stat().st_size / 1024 / 1024:.1f} MB, "
                    "retrying with smaller duration"
                )
            chunk_path.unlink()
            half_ms = chunk_ms // 2
            sub_step = half_ms - CHUNK_OVERLAP_MS
            sub_start, sub_idx = start, 0
            while sub_start < end:
                sub_end = min(sub_start + half_ms, end)
                sub_path = temp_dir / f"chunk_{i:03d}_{sub_idx:02d}.mp3"
                _export_chunk(audio[sub_start:sub_end], sub_path)
                chunk_paths.append(sub_path)
                sub_start += sub_step
                sub_idx += 1
        else:
            chunk_paths.append(chunk_path)

        if end >= total_ms:
            break
        i += 1

    if verbose:
        print(f"  Split into {len(chunk_paths)} chunks")
    return chunk_paths


def _export_chunk(segment: AudioSegment, path: Path) -> None:
    """Export an AudioSegment to *path* as 64 kbps mono MP3 at 16 kHz."""
    segment.export(
        str(path),
        format="mp3",
        parameters=["-ac", "1", "-ar", "16000", "-b:a", "64k"],
    )


def cleanup_chunks(chunk_paths: list[Path]) -> None:
    """Delete temporary chunk files and their parent directory."""
    for p in chunk_paths:
        try:
            p.unlink()
        except OSError:
            pass
    if chunk_paths:
        try:
            chunk_paths[0].parent.rmdir()
        except OSError:
            pass
