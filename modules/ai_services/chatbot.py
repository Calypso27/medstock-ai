from openai import OpenAI
from utils.config import XAI_BASE_URL, XAI_MODEL, get_xai_key

try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(**_):
        def decorator(fn): return fn
        return decorator


def _client():
    return OpenAI(api_key=get_xai_key(), base_url=XAI_BASE_URL)


@_traceable(name="ask_llm", run_type="llm")
def ask_grok(messages, system, temperature=0.7, max_tokens=1024):
    key = get_xai_key()
    if not key:
        return "Clé API manquante. Ajoutez GROQ_API_KEY dans le fichier .env"
    try:
        response = _client().chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Erreur API : {e}"


@_traceable(name="stream_llm", run_type="llm")
def stream_grok(messages, system, temperature=0.7):
    key = get_xai_key()
    if not key:
        yield "Clé API manquante. Ajoutez GROQ_API_KEY dans le fichier .env"
        return
    try:
        stream = _client().chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=temperature,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"Erreur : {e}"
