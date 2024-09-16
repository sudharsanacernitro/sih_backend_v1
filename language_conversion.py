from googletrans import Translator


language={'hi_IN':'hi',
            'en_US':'en'
    }

def translate_to_english(text):
    # Initialize the translator
    translator = Translator()
    # Translate the text to English
    translation = translator.translate(text, dest='en')

    # Return the translated text
    return translation.text

def english_to_other(lang,text):
    translator = Translator()

    lan=language[lang]

    translation = translator.translate(text, src='en', dest=lan)

    return translation.text




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
