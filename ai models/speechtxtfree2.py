
import tkinter as tk
import speech_recognition as sr
from threading import Thread

# Initialize recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300  # Adjust if needed
recognizer.pause_threshold = 1.0   # Longer pause detection

def recognize_speech():
    try:
        with sr.Microphone() as source:
            status_label.config(text="Adjusting for background noise...", fg="gray")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            print("🎙️ Listening...")
            status_label.config(text="Listening... Speak now.", fg="blue")

            # Listen until phrase_time_limit or short pause
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=20)

            print("⏳ Recognizing...")
            status_label.config(text="Recognizing...", fg="orange")

            text = recognizer.recognize_google(audio)

            print(f"✅ Recognized Text: {text}")
            result_label.config(text=f"📝 Text: {text}")
            status_label.config(text="Done", fg="green")

    except sr.WaitTimeoutError:
        result_label.config(text="⏱️ No speech detected.")
        status_label.config(text="Timeout", fg="red")
    except sr.UnknownValueError:
        result_label.config(text="🤷 Could not understand audio")
        status_label.config(text="Failed", fg="red")
    except sr.RequestError as e:
        result_label.config(text="❌ Google API Error")
        status_label.config(text="Error", fg="red")
        print(f"❌ Could not request results; {e}")

def on_button_click():
    Thread(target=recognize_speech).start()

# Setup GUI
root = tk.Tk()
root.title("🎤 Speech to Text")
root.geometry("500x250")
root.resizable(False, False)

tk.Label(root, text="Click the button and speak", font=("Arial", 14)).pack(pady=20)

button = tk.Button(root, text="Start Listening", command=on_button_click, font=("Arial", 12), bg="#4CAF50", fg="white")
button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=450, justify="center")
result_label.pack(pady=10)

status_label = tk.Label(root, text="", font=("Arial", 10))
status_label.pack(pady=5)

root.mainloop()
