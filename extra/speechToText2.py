# speechToText2.py

import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from google.cloud import speech_v1p1beta1 as speech

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    audio_path = "temp.webm"  # Save as webm
    audio_file.save(audio_path)

    client = speech.SpeechClient()

    with open(audio_path, 'rb') as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,  # ✅ match frontend recording
        sample_rate_hertz=48000,  # ✅ match opus standard
        language_code="en-US"
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return jsonify({"transcript": transcript or "No speech detected."})

if __name__ == '__main__':
    app.run(debug=True)
