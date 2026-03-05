from groq import Groq

DEFAULT_SUMMARY_MODEL = "llama-3.3-70b-versatile"

_LANG_NAMES: dict[str, str] = {
    "ru": "Russian", "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "tr": "Turkish",
    "uk": "Ukrainian", "pl": "Polish", "nl": "Dutch", "sv": "Swedish",
}


def summarize(
    text: str,
    api_key: str,
    language: str = "en",
    model: str = DEFAULT_SUMMARY_MODEL,
) -> str:
    """Summarize a meeting transcription using a Groq LLM.

    Produces a structured summary in the same language as the audio containing:
    an overview, key discussion points, decisions made, action items, and
    important deadlines.

    Args:
        text:     Full transcription text to summarize.
        api_key:  Groq API key.
        language: ISO-639-1 language code -- determines the language of the
                  summary output (e.g. ``"en"``, ``"ru"``).
        model:    Groq chat model ID to use.

    Returns:
        Structured summary as a plain string.
    """
    client = Groq(api_key=api_key)
    lang_name = _LANG_NAMES.get(language, language)

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
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the transcription to summarize:\n\n{text}"},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""
