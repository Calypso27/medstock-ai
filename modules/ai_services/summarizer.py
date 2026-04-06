from modules.ai_services.chatbot import ask_grok

_SYSTEM_MED = (
    "Tu es un assistant médical. Résume le texte suivant en exactement 3 phrases "
    "courtes et claires en français. Commence directement par le résumé."
)

_SYSTEM_FIN = (
    "Tu es un analyste financier. Résume la prédiction suivante en 3 phrases clés "
    "en français : cours cible, tendance, principal risque. "
    "Commence directement par le résumé."
)


def generate_medical_summary(text):
    if not text or not text.strip():
        return ""
    return ask_grok(
        messages=[{"role": "user", "content": text}],
        system=_SYSTEM_MED,
        temperature=0.3,
        max_tokens=300,
    )


def generate_financial_summary(text, ticker, model_name):
    if not text or not text.strip():
        return ""
    prompt = f"Prédiction {model_name} pour {ticker} :\n\n{text}"
    return ask_grok(
        messages=[{"role": "user", "content": prompt}],
        system=_SYSTEM_FIN,
        temperature=0.3,
        max_tokens=300,
    )
