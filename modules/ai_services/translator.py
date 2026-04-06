from modules.ai_services.chatbot import ask_grok

_SYSTEM = (
    "Du bist ein präziser medizinischer Übersetzer. "
    "Übersetze den folgenden Text vollständig und korrekt ins Deutsche. "
    "Antworte NUR mit der Übersetzung, ohne Erklärungen."
)


def translate_to_german(text):
    if not text or not text.strip():
        return ""
    return ask_grok(
        messages=[{"role": "user", "content": text}],
        system=_SYSTEM,
        temperature=0.2,
        max_tokens=2048,
    )
