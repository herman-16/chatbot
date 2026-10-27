import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Load model, vectorizer, dan dataset ---
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
df = pd.read_csv("dataset_chatbot.csv")  # pastikan kolom 'pertanyaan', 'intent', 'jawaban' ada

# Buat mapping intent → jawaban (mengambil jawaban pertama dari setiap intent sebagai fallback)
intent_to_answer = (
    df.drop_duplicates(subset=["intent"])
    .set_index("intent")["jawaban"]
    to_dict()
)

# Simpan vector corpus buat similarity search
corpus_vec = vectorizer.transform(df["pertanyaan"]).toarray()
corpus_texts = df["pertanyaan"].tolist()
corpus_answers = df["jawaban"].tolist()


def chatbot_predict(text: str, threshold=0.35):
  # Normalisasi dasar tanpa merusak fitur vectorizer (cukup lowercase dan strip)
  clean_text = text.lower().strip()

  # Penanganan khusus untuk sapaan (greetings) agar tidak nyasar ke pendaftaran
  greetings = ["hallo", "halo", "hai", "selamat pagi", "selamat siang", "pagi"]
  if clean_text in greetings:
    return (
        "Halo! Ada yang bisa saya bantu terkait informasi akademik Fakultas"
        " Hukum Unisba?",
        "lainnya",
        1.0,
    )

  # Transformasi teks langsung menggunakan vectorizer (sesuai saat training di Colab)
  X = vectorizer.transform([clean_text])
  pred = model.predict(X)[0]

  # Confidence dari Naive Bayes
  conf = 0.0
  if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)[0]
    conf = float(np.max(proba))

  # Jika confidence di atas threshold ATAU skor similarity sangat tinggi, gunakan intent
  # Jika tidak, cari kemiripan teks terdekat menggunakan Cosine Similarity
  sim_scores = cosine_similarity(X, corpus_vec)[0]
  max_sim = float(np.max(sim_scores))

  if conf >= threshold or max_sim >= 0.3:
    # Jika kemiripan kalimat sangat tinggi langsung ke pertanyaan spesifik di corpus
    if max_sim >= 0.5:
      idx = np.argmax(sim_scores)
      answer = corpus_answers[idx]
    else:
      # Gunakan jawaban berdasarkan intent
      answer = intent_to_answer.get(pred, corpus_answers[np.argmax(sim_scores)])
  else:
    # Fallback jika pertanyaan di luar konteks / tidak dikenali
    answer = (
        "Maaf, saya kurang memahami pertanyaan Anda. Silakan tanyakan hal"
        " lain seputar KRS, skripsi, jadwal ujian, atau administrasi Fakultas"
        " Hukum Unisba."
    )

  return answer, pred, conf


# --- Streamlit App ---
st.set_page_config(
    page_title="Chatbot Akademik FH Unisba", page_icon="🎓", layout="centered"
)
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

  with st.chat_message("assistant"):
    st.markdown(reply, unsafe_allow_html=True)

  st.session_state.messages.append({"role": "assistant", "content": reply})