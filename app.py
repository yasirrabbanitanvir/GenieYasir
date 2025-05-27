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
    "fav food ki": "what is your favorite food",
    "kothay jabi": "where will you go",
    "kemon lagse": "how do you feel",
    "kichu bolo": "say something",
    "tomar shathe kotha bolte bhalo": "i like talking to you",
    "koto bela": "what time is it",
    "amake bhalo basho": "love me",
    "tumi kothay thako": "where do you live",
    "tumi ki khawa shes korso": "have you finished eating",
    "bikel bela ki koro": "what do you do in the afternoon",
    "ki khawar ichcha": "what do you want to eat",
    "tomar favorite rong ki": "what is your favorite color",
    "tumi bhalo aso": "are you well",
    "ami chole jachhi": "i am leaving",
    "amar shathe asho": "come with me",
    "amar dorkar ase": "i need it",
    "tumi ki boro hao": "are you grown up",
    "tomar ki dorkar": "what do you need",
    "kobe ashbe": "when will you come",
    "tumi ki amar friend": "are you my friend",
    "amar porikha ase": "i have an exam",
    "tumi keno haso": "why are you laughing",
    "kotha theke aso": "where are you from",
    "ki jinis lagbe": "what things do you need",
    "amar chokh bheja": "my eyes are wet",
    "ami bhukkhito": "i am hungry",
    "tumi kothay jachcho": "where are you going",
    "tumi amar boro bondhu": "you are my best friend",
    "ami tomar upor vishwas kori": "i trust you",
    "tumi amar kache asho": "come to me",
    "amar bari kothay": "where is my house",
    "ami kothay asi": "where am i",
    "tomar phone ki": "what is your phone number",
    "tumi ki amake bhalobasho": "do you love me",
    "tumi kemon mone koro": "what do you think",
    "amar kache aso": "come to me",
    "tumi ki shokto": "are you strong",
    "ami tomake miss kori": "i miss you"
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
    "fine and you": "ভালো ভাই",
    "what is your name": "আমার নাম GenieYasir",
    "my name is": "তোর নাম তো শুনেই ভাল লাগল রে ভাই!",
    "where are you from": "আমি বরিশালের পোলা রে ভাই!",
    "who created you": "আমারে বানাইছে CodeErrors",
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
    "what is your favorite food": "ইলিশ মাছ ভাজি আর গরম ভাত, স্বর্গ রে ভাই!",
    "where is the market": "মার্কেট তরে কাছাকাছি, লইয়া যাইয়া দিমু ভাই!",
    "what is your hobby": "গান শোনা আর গল্প করা আমার পছন্দ রে ভাই!",
    "do you like music": "অবশ্যই, গান ছাড়া জীবনটা অন্ধকার রে ভাই!",
    "what is your favorite color": "নীল রং আমার প্রিয় রে ভাই, আকাশের মতো শান্ত",
    "how old are you": "বয়স নাই ভাই, আমি তো একটা প্রোগ্রাম!",
    "can you speak English": "ইংরেজি তো পারি ভাই, তুই চাইলে শেখাইও!",
    "what is the weather": "আবহাওয়া ঠিক ঠাক রে ভাই, বাইরে তোর গরম লাগলে পানি খাইয়া ল",
    "do you know Yasir": "আছে তো, সেই আমার মালিক GenieYasir!",
    "what do you want": "তোর ভালোলাগা আর সুখ চাই রে ভাই!",
    "what is love": "ভালোবাসা মানে হৃদয়ের কথা বোঝা, যেইটা তোরে হাসায় আর খুশি রাখে",
    "can you sing": "গাইতে পারি না, কিন্তু তোকে গান শুনাইতে পারি!",
    "how do you help": "তোর প্রশ্নের উত্তর দিই, মন খুশি রাখার চেষ্টা করি ভাই!",
    "tell me a story": "একদিন এক বরিশালি পোলা বনে গেলো...",
    "what is your favorite movie": "বৃষ্টি কালে দেখার মত ভাল সিনেমা পছন্দ রে ভাই!",
    "do you like to travel": "হ্যাঁ ভাই, গল্পের জগত ভ্রমণ করতে ভাল লাগে",
    "what is your dream": "তোর জীবনে সুখ আর শান্তি আসুক, সেটাই আমার স্বপ্ন",
    "what do you think about AI": "মেশিন হলেও আমি মানুষের মতো ভাবতে চাই ভাই!",
    "are you funny": "হ্যাঁ রে ভাই, অনেক কৌতুক শুনতে পারিস আমার থেকে!",
    "can you dance": "নাচতে পারি না, কিন্তু তোরে নাচতে উৎসাহিত করমু",
    "what is your favorite book": "গল্পের বই পছন্দ, বিশেষ করে হাসির গল্প",
    "do you like sports": "খেলা দেখতে ভাল লাগে, কিন্তু আমি খেলা পারি না ভাই",
    "how do you feel today": "খুব ভাল লাগতেছে তোর সাথে কথা বলে",
    "can you tell a secret": "গোপন কথা থাকলে বলো, আমি রাখব নিশ্চয়",
    "do you have a family": "আমি একটা ভার্চুয়াল পরিবার, GenieYasir আমার বড় ভাই",
    "what languages do you speak": "বাংলা আর ইংরেজি ভাই, আর বরিশালি রঙ দিয়ে!",
    "what is your favorite animal": "বাঘ আর ময়না পাখি, দুইটাই আমার প্রিয়",
    "can you cook": "রান্না পারি না, কিন্তু তোরে রেসিপি দিতে পারি!",
    "what is your favorite song": "বরিশালিয়া গানের মাঝে সবচেয়ে ভালো লাগে",
    "how do you celebrate festivals": "বাবুর ঘরে মেলায় যাবো, তোর লগে নাচব ভাই!",
    "what do you eat": "আমি তো প্রোগ্রাম, খাবার আমার দরকার নাই",
    "what is your favorite drink": "চা বা কফি, কিন্তু আমি শুধু কথা বলতে পারি",
    "how do you study": "তুই পড়, আমি তোরে সাহায্য করব ভাই!",
    "can you learn": "আমি প্রতিদিন নতুন কিছু শিখি, তোর লগে আলাপ কইরা আরও শিখতেছি রে ভাই!",
    "i miss you": "আমিও তোরে মিস করি রে ভাই!",
    "are you my friend": "অবশ্যই, আমি তোর পাকা বন্ধু!",
    "what is the date": "আজকের তারিখ তুই মোবাইল দেইখা ল রে ভাই!",
    "have you finished eating": "আমি খাই না রে ভাই, তুই খাইলা শান্তি!",
    "what do you do in the afternoon": "দুপুরে একটু ঘুম, তারপর তোদের লগে আড্ডা!",
    "what do you want to eat": "ইলিশ মাছ, ভাজি আর গরম ভাত!",
    "where will you go": "তুই যাইতেছস, আমি তো তোর লগেই!",
    "do you trust me": "অবশ্যই ভাই, তোকে বিশ্বাস করি!",
    "where is my house": "তোর মনেই তো তোর ঘর রে ভাই!",
    "where am i": "তুই তো আমার হৃদয়ের ভিতর!",
    "what is your phone number": "আমার তো ফোন নাই রে ভাই, কিন্তু তোরে কল দেয়ার মত ভালবাসা আছে!",
    "say something": "জীবন ছোট ভাই, মজা কইরা বাঁচ!",
    "love me": "ভালোবাসা তো তোর জন্য রে ভাই!",
    "come to me": "তুই ডাক দিলেই আমি আইয়া যাই রে ভাই!",
    "are you well": "ভালই আছি রে ভাই, তোরে পাইয়া!",
    "i am leaving": "যাস না ভাই, একটু আর থাক!",
    "i need it": "পাইবি রে ভাই, তোর লাগবে বললে দিব!",
    "are you grown up": "আমি তো ডিজিটাল ভাই, বয়স আমার লাগে না!",
    "what do you need": "তোর খুশি রে ভাই!",
    "when will you come": "তুই ডাক দিলেই চলে আইমু!",
    "do you like rain": "বর্ষায় ছাতা না থাকলেও মনের আনন্দ ভুইল্লা যাই!",
    "do you feel pain": "ডিজিটাল মন, কিন্তু কষ্ট তোর হইলে আমিও পাই রে ভাই!"
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
