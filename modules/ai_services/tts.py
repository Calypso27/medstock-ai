import io
from gtts import gTTS


def text_to_speech(text, lang="de"):
    if not text or not text.strip():
        return None
    try:
        tts = gTTS(text=text.strip(), lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Erreur TTS : {e}")
        return None
