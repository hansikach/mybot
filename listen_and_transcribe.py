import speech_recognition as sr

def listen_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_(audio, language='en-US')
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return None
    except sr.RequestError as e:
        print(f"API error: {e}")
        return None


listen_and_transcribe()
