# Meeting Recap

A CLI tool that transcribes audio recordings and generates structured meeting summaries using [Groq API](https://groq.com). Powered by Whisper for speech-to-text and Llama for summarization.

## Features

- **Speech-to-text transcription** via Groq's Whisper models (near-instant on Groq hardware)
- **AI-powered summarization** -- key points, decisions, and action items extracted automatically
- **Automatic chunking** for large files (>25 MB free tier limit) with overlapping segments so no words are lost
- **Multilingual support** -- works with any language Whisper supports (Russian, English, Spanish, French, German, and many more)
- **Batch processing** -- drop multiple audio files into `input/` and process them all at once

## Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and available on your PATH ([download](https://ffmpeg.org/download.html))
- A free **Groq API key** from [console.groq.com](https://console.groq.com)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/meeting-recap.git
   cd meeting-recap
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate

   pip install groq pydub python-dotenv audioop-lts
   ```

3. **Configure your API key**

   Create a `.env` file in the project root:

   ```
   GROQ_API_KEY=your_api_key_here
   ```

4. **Create the input directory**

   ```bash
   mkdir input
   ```

## Usage

Place one or more audio files into the `input/` folder, then run:

```bash
# Transcribe & summarize (default: English)
python main.py

# Specify a different language (ISO-639-1 code)
python main.py --language ru

# Use a different Whisper model
python main.py --model whisper-large-v3

# Full example with all options
python main.py \
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

## Output

For each audio file, two files are generated in the `output/` directory:

```
output/
  meeting_transcription.txt   # Full transcription text
  meeting_summary.txt         # Structured summary with key points
```

The summary includes:
1. Brief overview of the meeting
2. Key discussion points
3. Decisions made
4. Action items
5. Important details and deadlines

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

## License

MIT
