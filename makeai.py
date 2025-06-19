import os
import base64
import tempfile
import requests

from dotenv import load_dotenv
from flask import Flask, request, render_template_string, jsonify
from openai import OpenAI

load_dotenv()
# instantiate correctly
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

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
          rdr.onloadend = async ()=> {
            const base = rdr.result.split(',')[1];
            const res = await fetch('/talk',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio:base})});
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
    # save incoming webm
    tmp = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
    tmp.write(blob)
    tmp.flush()

    # 1) Google STT
    stt_url = f"https://speech.googleapis.com/v1p1beta1/speech:recognize?key={GOOGLE_API_KEY}"
    with open(tmp.name, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    stt_payload = {
        "config":{"encoding":"WEBM_OPUS","languageCode":"en-US"},
        "audio": {"content": audio_b64}
    }
    tr = requests.post(stt_url, json=stt_payload).json()
    transcript = tr.get("results", [{}])[0].get("alternatives", [{}])[0].get("transcript","")

    # 2) GPT
    resp = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role":"user","content":transcript}]
    )
    reply = resp.choices[0].message.content

    # 3) Google TTS
    tts_url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"
    tts_payload = {
      "input":{"text":reply},
      "voice":{"languageCode":"en-US","name":"en-US-Wavenet-F"},
      "audioConfig":{"audioEncoding":"MP3"}
    }
    tts = requests.post(tts_url, json=tts_payload).json().get("audioContent","")

    return jsonify(transcript=transcript or "—", audio=tts)

if __name__=='__main__':
    app.run(debug=True)
