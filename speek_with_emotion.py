import pyttsx3
from textblob import TextBlob

def speak_with_emotion(text):
    engine = pyttsx3.init()

    voices = engine.getProperty('voices')
    for voice in voices:
        print(f"{voice.id} | {voice.name} | {voice.languages}")

    sentiment = TextBlob(text).sentiment.polarity

    # Adjust pitch and rate based on sentiment
    if sentiment > 0.2:
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
    elif sentiment < -0.2:
        engine.setProperty('rate', 120)
        engine.setProperty('volume', 0.7)
    else:
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)

    #engine.setProperty('voice', '')  # Change to a suitable voice
    engine.say(text)
    engine.runAndWait()

speak_with_emotion("I'm thrilled to be working on this project!")
