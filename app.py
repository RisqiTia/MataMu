from flask import Flask, request, jsonify
import pickle
import json
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
import os

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Download resource NLTK (untuk pertama kali saja)
nltk.download('punkt')
nltk.download('wordnet')

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# Fungsi preprocessing input pengguna
def clean_text(text):
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w.lower()) for w in tokens])

# Load model dan intents
model = pickle.load(open("chatbot_model.pkl", "rb"))

with open("intents_label.json", encoding="utf-8") as f:
    intents = json.load(f)

# Endpoint utama untuk chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Mohon masukkan pesan yang valid."})

    cleaned_message = clean_text(user_message)
    predicted_tag = model.predict([cleaned_message])[0]

    for intent in intents['intents']:
        tag = intent.get("tag")
        if isinstance(tag, list):
            tag = tag[0]
        if tag == predicted_tag:
            return jsonify({
                "response": random.choice(intent['responses'])
            })

    return jsonify({"response": "Maaf, saya tidak mengerti maksud Anda."})

# Endpoint opsional untuk pengecekan server
@app.route("/")
def home():
    return "<h3>✅ Chatbot EyeBot Flask aktif. Gunakan endpoint POST /chat</h3>"

# Jalankan server (untuk Render atau platform deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render akan inject $PORT
    app.run(debug=False, host="0.0.0.0", port=port)
