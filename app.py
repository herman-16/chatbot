import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Load basis FAQ dan siapkan pencarian berdasarkan dataset aktif ---
df = pd.read_csv("dataset_chatbot.csv")
required_columns = {"pertanyaan", "jawaban", "intent", "sumber"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
  raise ValueError(
      "Kolom dataset chatbot tidak lengkap: " + ", ".join(sorted(missing_columns))
  )

if df[["pertanyaan", "jawaban", "intent"]].isna().any().any():
  raise ValueError("Dataset chatbot memiliki pertanyaan, jawaban, atau intent kosong.")

stemmer = StemmerFactory().create_stemmer()


def preprocess(text: str) -> str:
  normalized = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
  return " ".join(stemmer.stem(token) for token in normalized.split())


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)
corpus_vec = vectorizer.fit_transform(df["pertanyaan"].map(preprocess))
corpus_answers = df["jawaban"].astype(str).tolist()
corpus_intents = df["intent"].astype(str).tolist()
corpus_sources = df["sumber"].fillna("").astype(str).tolist()

FALLBACK_ANSWER = (
    "Maaf, saya belum menemukan jawaban yang cukup relevan di basis informasi. "
    "Coba perjelas pertanyaan dengan menyebutkan topik atau istilah akademiknya."
)


def chatbot_predict(text: str, threshold=0.40, min_margin=0.08):
  clean_text = preprocess(text)
  greetings = {"hallo", "halo", "hai", "selamat pagi", "selamat siang", "pagi"}
  if clean_text in greetings:
    return (
        "Halo! Ada yang bisa saya bantu terkait informasi akademik Fakultas"
        " Hukum Unisba?",
        "lainnya",
        1.0,
    )

  query_vec = vectorizer.transform([clean_text])
  if query_vec.nnz == 0:
    return FALLBACK_ANSWER, "lainnya", 0.0

  similarity_scores = cosine_similarity(query_vec, corpus_vec).ravel()
  best_idx = int(similarity_scores.argmax())
  best_score = float(similarity_scores[best_idx])

  best_score_by_answer = {}
  for answer, score in zip(corpus_answers, similarity_scores):
    best_score_by_answer[answer] = max(
        best_score_by_answer.get(answer, 0.0), float(score)
    )

  distinct_answer_scores = sorted(best_score_by_answer.values(), reverse=True)
  second_score = distinct_answer_scores[1] if len(distinct_answer_scores) > 1 else 0.0
  if best_score < threshold or best_score - second_score < min_margin:
    return FALLBACK_ANSWER, corpus_intents[best_idx], best_score

  answer = corpus_answers[best_idx]
  source = corpus_sources[best_idx]
  if source:
    answer = f"{answer}\n\nSumber: {source}"

  return answer, corpus_intents[best_idx], best_score


# --- Streamlit App ---
st.set_page_config(
    page_title="Chatbot Akademik FH Unisba", page_icon="🎓", layout="centered"
)
st.title("Tanyakan tentang Akademik Fakultas Hukum Unisba disini🎓")

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