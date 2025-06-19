import tkinter as tk
import speech_recognition as sr
from threading import Thread

# Initialize recognizer
recognizer = sr.Recognizer()

def recognize_speech():
    try:
        with sr.Microphone() as source:
            print("🎙️ Listening...")
            status_label.config(text="Listening... Speak now.", fg="blue")
            audio = recognizer.listen(source)

            print("⏳ Recognizing...")
            status_label.config(text="Recognizing...", fg="orange")
            text = recognizer.recognize_google(audio)

            print(f"✅ Recognized Text: {text}")
            result_label.config(text=f"📝 Text: {text}")
            status_label.config(text="Done", fg="green")
    except sr.UnknownValueError:
        result_label.config(text="Could not understand audio")
        status_label.config(text="Failed", fg="red")
        print("❌ Could not understand audio")
    except sr.RequestError as e:
        result_label.config(text="Google API Error")
        status_label.config(text="Error", fg="red")
        print(f"❌ Could not request results; {e}")

def on_button_click():
    # Run recognition in a separate thread so GUI doesn't freeze
    Thread(target=recognize_speech).start()

# Setup GUI
root = tk.Tk()
root.title("🎤 Speech to Text")
root.geometry("500x250")
root.resizable(False, False)

tk.Label(root, text="Click the button and speak", font=("Arial", 14)).pack(pady=20)

button = tk.Button(root, text="Start Listening", command=on_button_click, font=("Arial", 12), bg="#4CAF50", fg="white")
button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

status_label = tk.Label(root, text="", font=("Arial", 10))
status_label.pack(pady=5)

root.mainloop()
