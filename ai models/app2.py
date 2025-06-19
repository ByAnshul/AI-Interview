import os
import base64
import tempfile
import requests
from dotenv import load_dotenv
from flask import Flask, request, render_template_string, jsonify
import openai

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY2"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

# HTML interface
HTML = """
<!doctype html>
<html>
  <body>
    <button onclick="start()">Start</button>
    <button onclick="stop()">Stop</button>
    <p>Transcript: <span id="txt"></span></p>
    <audio id="player" controls></audio>
    <script>
      let rec, chunks = [];
      async function start(){
        const s = await navigator.mediaDevices.getUserMedia({audio:true});
        rec = new MediaRecorder(s);
        rec.ondataavailable = e=>chunks.push(e.data);
        rec.start();
      }
      function stop(){
        rec.onstop = async ()=>{
          const blob = new Blob(chunks, { type:'audio/webm' });
          const rdr = new FileReader();
          rdr.onloadend = async ()=>{
            const base = rdr.result.split(',')[1];
            const res = await fetch('/talk',{
              method:'POST',
              headers:{'Content-Type':'application/json'},
              body:JSON.stringify({audio:base})
            });
            const js = await res.json();
            document.getElementById('txt').innerText = js.transcript;
            document.getElementById('player').src = 'data:audio/mp3;base64,' + js.audio;
          };
          rdr.readAsDataURL(blob);
        };
        rec.stop();
      }
    </script>
  </body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/talk', methods=['POST'])
def talk():
    data = request.get_json()
    blob = base64.b64decode(data['audio'])

    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp.write(blob)
        tmp.flush()
        tmp_path = tmp.name

    # Step 1: Transcribe with Whisper
    try:
        with open(tmp_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        transcript = response.text
    except Exception as e:
        return jsonify(transcript=f"Transcription Error: {e}", audio="")

    # Step 2: Chat with GPT-4.1-mini Interviewer
    try:
        chat_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional AI interviewer for a Software Engineering role at a top tech company. "
                        "Your goal is to evaluate the candidate’s communication, technical understanding, and problem-solving skills. "
                        "Ask questions one at a time, and wait for the candidate's response. Keep a professional tone."
                    )
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ]
        )
        reply = chat_response.choices[0].message.content
    except Exception as e:
        reply = "Sorry, I couldn't process your response."

    # Step 3: Google Text-to-Speech
    try:
        tts_url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"
        tts_payload = {
            "input": {"text": reply},
            "voice": {"languageCode": "en-US", "name": "en-US-Wavenet-F"},
            "audioConfig": {"audioEncoding": "MP3"}
        }
        tts = requests.post(tts_url, json=tts_payload).json()
        audio_content = tts.get("audioContent", "")
    except Exception as e:
        audio_content = ""

    return jsonify(transcript=transcript or "—", audio=audio_content)

if __name__ == '__main__':
    app.run(debug=True)
