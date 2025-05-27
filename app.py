from flask import Flask, render_template, request, jsonify
import random
import json

app = Flask(__name__)

barisal_dict = {
    "hello": "হ্যালো রে ভাই! কেমন আছস?",
    "hi": "হাই রে ভাই! কি খবর তোর?",
    "how are you": "তুই কেমন আছস রে ভাই?",
    "what is your name": "তোর নাম কী রে ভাই?",
    "my name is": "তোর নাম তো শুনেই ভাল লাগল রে ভাই!",
    "where are you from": "আমি বরিশালের পোলা রে ভাই!",
    "who created you": "আমারে বানাইছে তুইই তো রে!",
    "thank you": "তোকে অশেষ ধন্যবাদ রে ভাই!",
    "thanks": "ধন্যবাদ রে ভাই!",
    "bye": "আচ্ছা ভাই, দেখা হইবো পরে!",
    "goodbye": "আল্লাহ হাফেজ রে ভাই!",
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get():
    user_input = request.json.get("msg", "").lower()
    if "joke" in user_input or "কৌতুক" in user_input:
        joke = random.choice(jokes)
        reply = joke.get("bangla", "মজার কৌতুক এখন নেই, পরে চেষ্টা করো!")
    else:
        for key in barisal_dict:
            if key in user_input:
                reply = barisal_dict[key]
                break
        else:
            reply = "দুঃখিত ভাই, আমি বুঝতে পারছি না, আরেকটু স্পষ্ট কইরা কইবা!"
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
