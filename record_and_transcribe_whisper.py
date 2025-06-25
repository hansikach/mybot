import pyaudio
import numpy as np
import soundfile as sf
import whisper
import tempfile
import threading
import time

# Whisper model (choose size: tiny, base, small, medium, large)
model = whisper.load_model("base")

# Audio recording parameters
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DURATION = 5  # seconds per chunk


def record_and_transcribe():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening and transcribing...")
    try:
        while True:
            frames = []
            start = time.time()
            while time.time() - start < DURATION:
                data = stream.read(CHUNK)
                frames.append(np.frombuffer(data, dtype=np.int16))

            audio_np = np.hstack(frames).astype(np.float32) / 32768.0
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(f.name, audio_np, RATE)
                result = model.transcribe(
                    temp_path,  # str | np.ndarray | torch.Tensor
                    language="en",  # Optional: ISO 639-1 code like 'en', 'hi', 'te', etc.
                    task="transcribe",  # or "translate" (to English)
                    verbose=False,  # Set True to print decoding progress
                    word_timestamps=True,  # Get word-level timestamps
                    condition_on_previous_text=True,  # Use previous segment as context
                    initial_prompt=None,  # Optional: prime the model with a prompt
                    temperature=0.5,  # Sampling temperature (0.0 = deterministic)
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6
                )

                print(" ", result["text"])
    except KeyboardInterrupt:
        print("Stopped.")
        stream.stop_stream()
        stream.close()
        p.terminate()


record_and_transcribe()
