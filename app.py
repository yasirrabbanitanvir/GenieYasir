from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import json
import numpy as np

app = Flask(__name__)
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

banglish_map = {
    "kmn acho": "how are you",
    "ki korteso": "what are you doing",
    "tor nam ki": "what is your name",
    "amar nam": "my name is",
    "ami dukhi": "i am sad",
    "ami valo asi": "i am happy",
    "valobashi": "do you love me",
    "tmi kothay thako": "where are you from",
    "ke banayse": "who created you",
    "thank you": "thank you",
    "biday": "bye",
    "joke dao": "tell me a joke",
    "door kholo": "open the door",
    "help lagbe": "i need help",
    "help koro": "can you help me",
    "fav food ki": "what is your favorite food"
}

def preprocess_banglish(text):
    text = text.lower()
    for banglish, english in banglish_map.items():
        if banglish in text:
            return english
    return text

barisal_dict = {
    "hello": "হ্যালো রে ভাই! কেমন আছস?",
    "how are you": "তুই কেমন আছস রে ভাই?",
    "what is your name": "তোর নাম কী রে ভাই?",
    "my name is": "তোর নাম তো শুনেই ভাল লাগল রে ভাই!",
    "where are you from": "আমি বরিশালের পোলা রে ভাই!",
    "who created you": "আমারে বানাইছে তুইই তো রে!",
    "thank you": "তোকে অশেষ ধন্যবাদ রে ভাই!",
    "bye": "আচ্ছা ভাই, দেখা হইবো পরে!",
    "what can you do": "আমি তোরে গল্প শুনাইতে পারি, কৌতুক বলতে পারি, আর মজা করাইতে পারি রে!",
    "tell me a joke": "তোরে একটা কৌতুক শুনাই, মন খুশি হইয়া যাইবো!",
    "i am sad": "ক্যান ভাই? মন খারাপ কইরো না, লাইফে সব ঠিক হইয়া যাইবো!",
    "i am happy": "ভাল লাগল শুনে ভাই! খুশিতে নাচ শুরু করো!",
    "are you real": "আমি রে ভাই, বাস্তব তো না, কিন্তু তোরে লাইফে সহায় করবার আসছি!",
    "what is the time": "ঘড়ি নাই রে ভাই, তুই মোবাইল দেইখা ল!",
    "what are you doing": "তোর লগে আলাপ কইরা আনন্দে আছি রে ভাই!",
    "do you love me": "তোরে না ভালবাসলে আর কারে করমু রে ভাই?",
    "open the door": "দুয়ার খুলাইতেছি ভাই, একটু ধৈর্য ধইরা থাক।",
    "can you help me": "অবশ্যই পারি ভাই! বল কী লাগবো?",
    "i need help": "বল রে ভাই, কি সমস্যা? আমি আছি তো!",
    "what is your favorite food": "ইলিশ মাছ ভাজি আর গরম ভাত, স্বর্গ রে ভাই!"
}

with open('jokes.json', 'r', encoding='utf-8') as f:
    jokes = json.load(f)

known_phrases = list(barisal_dict.keys())
known_embeddings = model.encode(known_phrases)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get():
    user_input = request.json.get("msg", "").strip()
    user_input_processed = preprocess_banglish(user_input)
    if any(keyword in user_input_processed.lower() for keyword in ["joke", "কৌতুক"]):
        joke = random.choice(jokes)
        reply = joke.get("bangla", "মজার কৌতুক এখন নেই, পরে চেষ্টা করো!")
    else:
        input_embedding = model.encode([user_input_processed])
        similarities = cosine_similarity(input_embedding, known_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        if similarities[best_match_idx] >= 0.4:
            matched_key = known_phrases[best_match_idx]
            reply = barisal_dict[matched_key]
        else:
            reply = "দুঃখিত ভাই, আমি বুঝতে পারছি না, আরেকটু স্পষ্ট কইরা কইবা!"
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)