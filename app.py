import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK & Sastrawi (stopwords + stemmer) ---
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# Pastikan stopwords ada
try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- Load model, vectorizer, dan dataset ---
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
df = pd.read_csv("dataset_chatbot.csv")   # pastikan ada kolom 'pertanyaan','intent','jawaban'

# Buat mapping intent → jawaban
intent_to_answer = dict(zip(df['intent'], df['jawaban']))

# Simpan vector corpus buat similarity search
corpus_vec = vectorizer.transform(df['pertanyaan']).toarray()
corpus_texts = df['pertanyaan'].tolist()
corpus_answers = df['jawaban'].tolist()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def chatbot_predict(text: str, threshold=0.4):
    clean = preprocess(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]

    # Confidence dari Naive Bayes
    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        idx = np.argsort(proba)[::-1][:3]
        conf = float(proba[idx[0]])

    if conf is not None and conf >= threshold:
        # Kalau confidence tinggi → pakai jawaban intent
        answer = intent_to_answer.get(pred, "Maaf, saya belum punya jawaban.")
    else:
        # Kalau confidence rendah → cari jawaban paling mirip dengan cosine similarity
        sim_scores = cosine_similarity(X.toarray(), corpus_vec)[0]
        idx = np.argmax(sim_scores)
        answer = corpus_answers[idx]

    return answer, pred, conf

# --- Streamlit App ---
st.set_page_config(page_title="Chatbot Akademik FH Unisba", page_icon="🎓", layout="centered")
st.title("🤖 Chatbot Layanan Akademik - Fakultas Hukum Unisba")

# Sidebar tools
with st.sidebar:
    st.header("⚙️ Opsi")
    if st.button("Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()

# State untuk riwayat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input chat
prompt = st.chat_input("Ketik pertanyaan Anda di sini...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply, intent, conf = chatbot_predict(prompt)

    # Jawaban tanpa metadata intent/confidence
    bot_text = reply

    with st.chat_message("assistant"):
        st.markdown(bot_text, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": bot_text})
