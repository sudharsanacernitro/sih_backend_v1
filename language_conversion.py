from deep_translator import GoogleTranslator

language = {
    'hi_IN': 'hi',
    'en_US': 'en',
    'ta_IN':'ta'
}

def translate_to_english(text):
    try:
        # Translate the text to English
        translation = GoogleTranslator(source='auto', target='en').translate(text)
        # Return the translated text
        print(translation)
        return translation
    except Exception as e:
        print(e)

def english_to_other(lang, text):
    try:
        lan = language[lang]
        # Translate from English to the target language
        translation = GoogleTranslator(source='en', target=lan).translate(text)
        return translation
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # Example text in various languages
    texts = [
        "جا کر ہمارے مالک کے لئے کچھ پانی خری دو"
    ]

    # Translate each text to English
    for text in texts:
        translated_text = translate_to_english(text)
        print(f"Original Text: {text}")
        print(f"Translated Text: {translated_text}\n")
