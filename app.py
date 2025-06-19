import os
from flask import Flask, request, render_template, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
import sqlite3
import bcrypt
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
import base64



import tempfile
import base64
import requests
from dotenv import load_dotenv
import openai
# Load .env variables
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY2"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("openai api key:", os.getenv("OPENAI_API_KEY2"))


# --- App Configuration ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = os.path.join('static', 'uploads')  # Correct path
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Utility: Check allowed file type ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- DB Connection Helper ---
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 1. Homepage Access (Unauthenticated or Authenticated)
# ============================================================

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('profile'))
    return render_template('home.html')


# ============================================================
# 2. Login Page Access and Submission
# ============================================================

@app.route('/loginPage', methods=['GET'])
def login_page():
    return render_template('loginPage.html')


@app.route('/loginPage', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password').encode('utf-8')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password, user['password']):
        session['user'] = email
        session['user_id'] = user['id']
        session['first_name'] = user['first_name']
        return redirect(url_for('profile'))
    else:
        flash('Invalid credentials')
        return redirect(url_for('login_page'))


# ============================================================
# 3. Signup / Register New User
# ============================================================

@app.route('/signup', methods=['POST'])
def signup():
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    email = request.form.get('email')
    password = request.form.get('password').encode('utf-8')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()

    if user:
        conn.close()
        flash('User already exists!')
        return redirect(url_for('login_page'))

    hashed_pw = bcrypt.hashpw(password, bcrypt.gensalt())
    cursor.execute(
        'INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)',
        (first_name, last_name, email, hashed_pw)
    )

    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    session['user'] = email
    session['user_id'] = user_id
    session['first_name'] = first_name

    return redirect(url_for('login_page'))


# ============================================================
# 4. User Logout
# ============================================================

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)  # ✅ Add this

    flash('You have been logged out.')
    return redirect(url_for('home'))


# ============================================================
# 5. Profile Page (User Dashboard)
# ============================================================

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect('/')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()

    return render_template('profile.html',
    user=user,
    name=f"{user['first_name']} {user['last_name']}",
    pronouns=user['pronouns'],
    bio=user['bio'],
    location=user['location'],
    skills=user['skills'],
    resume=user['resume_path']
)



# ============================================================
# 6. Update Profile Info + Resume Upload
# ============================================================

# === PDF to Text Utility ===
def extract_text_from_uploaded_resume(upload_folder, uploaded_filename):
    """
    Extract text from an uploaded resume PDF file.
    """
    full_path = os.path.join(upload_folder, uploaded_filename).replace("\\", "/")
    try:
        with fitz.open(full_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Utility to extract text from PDF
def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text





@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect('/')
    data = {}
    resume_path = None 
    # Handle resume upload
    resume_file = request.files.get('resume')
    if resume_file and allowed_file(resume_file.filename):
        filename = secure_filename(resume_file.filename)
        resume_save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\", "/")
        resume_file.save(resume_save_path)

        resume_path = os.path.join('uploads', filename).replace("\\", "/")
        data['resume_path'] = resume_path

        # ✅ Extract text using helper
        pdf_text = extract_text_from_uploaded_resume(app.config['UPLOAD_FOLDER'], filename)
        if pdf_text:
            print("✅ Extracted Resume Text (first 300 chars):")
            print(pdf_text[:300])  # Optional truncate
            data['resume_text'] = pdf_text
        else:
            print("⚠️ Failed to extract text from resume.")

    # Handle photo upload
    photo_path = None
    photo_file = request.files.get('photo')
    if photo_file and allowed_file(photo_file.filename):
        filename = secure_filename(photo_file.filename)
        photo_full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo_file.save(photo_full_path)
        photo_path = os.path.join('uploads', filename).replace("\\", "/")
        print(f"Saved photo_path: {photo_path}")


    # Collect fields
    data = {
        'first_name': request.form.get('first_name'),
        'last_name': request.form.get('last_name'),
        'bio': request.form.get('bio'),
        'location': request.form.get('location'),
        'pronouns': request.form.get('pronouns'),
        'website': request.form.get('website'),
        'calendar_link': request.form.get('calendar'),
        'twitter': request.form.get('twitter'),
        'linkedin': request.form.get('linkedin'),
        'github': request.form.get('github'),
        'instagram': request.form.get('instagram'),
        'skills': request.form.get('skills'),
    }

    if resume_path:
        data['resume_path'] = resume_path
    if photo_path:
        data['photo_path'] = photo_path

    # Build update query dynamically
    placeholders = ', '.join([f"{key} = ?" for key in data if data[key] is not None])
    values = [data[key] for key in data if data[key] is not None]
    values.append(session['user_id'])

    query = f"UPDATE users SET {placeholders} WHERE id = ?"

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()
    conn.close()

    flash("Profile updated successfully!")
    return redirect(url_for('profile'))

# ============================================================
# 7. Profile Settings Page (Edit Form)
# ============================================================

@app.route('/profile_settings')
def profile_settings():
    if 'user_id' not in session:
        return redirect('/')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()

    return render_template('profile_setting.html', user=user)


# ============================================================
# 8. Job Listings Page + Job Detail Pages
# ============================================================

@app.route('/jobs')
def jobs():
    return render_template('jobs.html', user_id=session.get('user_id'))
   
        # return redirect(url_for('home'))  # Optional auth
        

@app.route('/jobs/<int:job_id>')
def job_detail(job_id):
    try:
        return render_template(f'job{job_id}.html', user_id=session.get('user_id'))
    except:
        return "Job not found", 404

# ============================================================
#ACTUAL CODE
# ============================================================
@app.route('/talk', methods=['POST'])
def talk():
    #if 'user_id' not in session:
        #return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    blob = base64.b64decode(data['audio'])

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp.write(blob)
        tmp.flush()
        tmp_path = tmp.name

    # Step 1: Transcribe audio with Whisper
    try:
        with open(tmp_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        transcript = response.text
    except Exception as e:
        return jsonify(transcript=f"Transcription Error: {e}", audio="")

    # Step 2: Generate GPT Interview reply
    try:
        chat_response = client.chat.completions.create(
            model="gpt-4.1-mini",  # or "gpt-4.1-mini" if available
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

    # Step 3: Convert reply to speech using Google TTS
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


@app.route('/price')
def pricing():
    return render_template('price.html')


# ============================================================
# Take to Interview Page
# ============================================================

@app.route('/interview')
def interview():
    if 'user_id' not in session:
        return redirect('/loginPage')
    return render_template('interview.html')







if __name__ == '__main__':
    app.run(debug=True)
