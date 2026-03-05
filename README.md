# Meeting Recap

A CLI tool **and Python library** that transcribes audio recordings and generates structured meeting summaries using [Groq API](https://groq.com). Powered by Whisper for speech-to-text and Llama for summarization.

## Features

- **Speech-to-text transcription** via Groq's Whisper models (near-instant on Groq hardware)
- **AI-powered summarization** -- key points, decisions, and action items extracted automatically
- **Automatic chunking** for large files (>25 MB free tier limit) with overlapping segments so no words are lost
- **Multilingual support** -- works with any language Whisper supports (Russian, English, Spanish, French, German, and many more)
- **Batch processing** -- drop multiple audio files into `input/` and process them all at once
- **Importable library** -- use it in your own Python projects with a clean API

## Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and available on your PATH ([download](https://ffmpeg.org/download.html))
- A free **Groq API key** from [console.groq.com](https://console.groq.com)

---

## Install from PyPI

```bash
pip install meeting-recap
```

> **Python 3.13+** also requires the `audioop-lts` shim (automatically installed as a dependency).

---

## CLI Usage

### From a cloned repo

```bash
   git clone https://github.com/talhatek/meeting-recap.git
cd meeting-recap
python -m venv venv && venv\Scripts\activate  # Windows
pip install -e .
```

Create a `.env` file with your API key:

```
GROQ_API_KEY=your_api_key_here
```

Place one or more audio files into the `input/` folder, then run:

```bash
# Transcribe & summarize (default: English)
meeting-recap
# or
python main.py

# Specify a different language (ISO-639-1 code)
meeting-recap --language ru

# Use a different Whisper model
meeting-recap --model whisper-large-v3

# Full example with all options
meeting-recap \
  --language en \
  --model whisper-large-v3-turbo \
  --summary-model llama-3.3-70b-versatile \
  --input-dir input \
  --output-dir output
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-l`, `--language` | `en` | ISO-639-1 language code of the audio |
| `-m`, `--model` | `whisper-large-v3-turbo` | Whisper model for transcription |
| `--summary-model` | `llama-3.3-70b-versatile` | LLM model for summarization |
| `--input-dir` | `input` | Directory containing audio files |
| `--output-dir` | `output` | Directory for output files |

### Supported Audio Formats

`flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm`

### Output

For each audio file, two files are generated in the `output/` directory:

```
output/
  meeting_transcription.txt   # Full transcription text
  meeting_summary.txt         # Structured summary with key points
```

---

## Library Usage

Install and import in any Python project:

```bash
pip install meeting-recap
```

### Full pipeline (transcribe + summarize)

```python
from meeting_recap import process

result = process("meeting.mp3", api_key="gsk_...")
print(result.transcription)
print(result.summary)
```

### Transcription only

```python
from meeting_recap import transcribe

text = transcribe("meeting.mp3", api_key="gsk_...", language="ru")
print(text)
```

### Summarization only

```python
from meeting_recap import summarize

summary = summarize(text, api_key="gsk_...", language="ru")
print(summary)
```

### Full API reference

```python
from meeting_recap import process, transcribe, summarize, RecapResult

# process() -- full pipeline
result: RecapResult = process(
    audio_path="meeting.mp3",
    api_key="gsk_...",
    language="en",                        # ISO-639-1 language code
    model="whisper-large-v3-turbo",       # Whisper model
    summary_model="llama-3.3-70b-versatile",  # LLM model
    verbose=True,                         # print progress
)
result.audio_path       # pathlib.Path to the source file
result.transcription    # full transcription string
result.summary          # structured summary string

# transcribe() -- speech-to-text only
text = transcribe(
    audio_path="meeting.mp3",
    api_key="gsk_...",
    language="en",
    model="whisper-large-v3-turbo",
    verbose=True,
)

# summarize() -- summarize existing text
summary = summarize(
    text="...",
    api_key="gsk_...",
    language="en",
    model="llama-3.3-70b-versatile",
)
```

---

## How It Works

```
input/*.mp3
    |
    v
[File size check]
    |
    +--> <= 25 MB --> Groq Whisper API --> Transcription
    |
    +--> > 25 MB ---> Split into overlapping chunks (pydub + ffmpeg)
                        |
                        +--> Chunk 1 --> Groq Whisper API --+
                        +--> Chunk 2 --> Groq Whisper API --+--> Merge --> Transcription
                        +--> Chunk N --> Groq Whisper API --+
                                                                    |
                                                                    v
                                                            Groq LLM (summarize)
                                                                    |
                                                                    v
                                                            output/*_summary.txt
                                                            output/*_transcription.txt
```

**Chunking details:**
- Files over 25 MB are split into segments that fit within Groq's free tier upload limit
- 5-second overlap between chunks prevents words from being cut off at boundaries
- Chunks are exported as 64 kbps mono MP3 at 16 kHz (optimal for speech recognition)
- Temporary chunk files are automatically cleaned up after processing

---

## Supported Languages

Any language supported by OpenAI Whisper. Common examples:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `ru` | Russian | `en` | English |
| `es` | Spanish | `fr` | French |
| `de` | German | `it` | Italian |
| `pt` | Portuguese | `zh` | Chinese |
| `ja` | Japanese | `ko` | Korean |
| `uk` | Ukrainian | `tr` | Turkish |
| `ar` | Arabic | `pl` | Polish |

Full list: [Whisper supported languages](https://github.com/openai/whisper#available-models-and-languages)

---

## License

MIT
