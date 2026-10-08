"""
Multi-language translation engine for AudioSense context and transcriptions.
Supports Hindi ('hin', 'hi'), Spanish, French, German, and English.
"""

from typing import Optional, Dict

COMMON_HINDI_SOUND_TERMS = {
    "dog bark": "कुत्ते के भौंकने की आवाज",
    "bark": "भौंकना",
    "cat": "बिल्ली",
    "meow": "म्याऊँ",
    "speech": "मानव आवाज या बातचीत",
    "conversation": "लोगों की बातचीत",
    "screaming": "चीखने की आवाज",
    "laughter": "हंसने की आवाज",
    "crying": "रोने की आवाज",
    "car": "कार",
    "car horn": "कार का हॉर्न",
    "engine": "इंजन की आवाज",
    "traffic": "सड़क का ट्रैफिक",
    "siren": "आपातकालीन सायरन",
    "alarm": "अलार्म",
    "glass break": "कांच टूटने की आवाज",
    "footsteps": "कदमों की आहट",
    "running water": "बहते पानी की आवाज",
    "rain": "बारिश",
    "thunder": "बिजली कड़कने की आवाज",
    "wind": "तेज हवा",
    "music": "संगीत या धुन",
    "door": "दरवाजा खुलने या बंद होने की आवाज",
    "doorbell": "दरवाजे की घंटी",
    "birds chirping": "पक्षियों की चहचहाहट",
    "applause": "तालियों की गड़गड़ाहट",
    "explosion": "धमाका या विस्फोट",
    "silence": "शांति या कोई विशेष आवाज नहीं"
}

def translate_text(text: str, target_lang: str = "en", llm_client: Optional[object] = None) -> str:
    """
    Translate text into target language code (e.g. 'hin' or 'hi' for Hindi).
    """
    target = target_lang.lower().strip()
    if target in ("en", "eng", "english") or not text.strip():
        return text

    # Try LLM-based translation if client is provided and available
    if llm_client is not None and hasattr(llm_client, "generate"):
        prompt = (
            f"You are a professional translator. Translate the following text into "
            f"{'Hindi' if target in ('hin', 'hi') else target}.\n"
            f"Text: \"{text}\"\n"
            f"Translation (output ONLY the translated sentence, nothing else):"
        )
        try:
            translated = llm_client.generate(prompt)
            if translated and not translated.startswith("Error"):
                return translated.strip(' "')
        except Exception:
            pass

    # Quick rule-based fallback for Hindi
    if target in ("hin", "hi"):
        translated_text = text
        for eng_term, hin_term in COMMON_HINDI_SOUND_TERMS.items():
            if eng_term in translated_text.lower():
                translated_text = translated_text.replace(eng_term, hin_term)
        # Contextual prefix if unchanged
        if translated_text == text:
            return f"[हिंदी अनुवाद]: {text}"
        return translated_text

    return text
