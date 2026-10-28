import io
import re
import pickle
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Chatbot Akademik FH UNISBA",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Chatbot Layanan Akademik")
st.caption("Fakultas Hukum Universitas Islam Bandung (UNISBA)")

st.write(
    "Alur sistem: pertanyaan mahasiswa → preprocessing → TF-IDF → "
    "Naive Bayes → prediksi intent → pemilihan jawaban dari dataset."
)


# =========================================================
# PREPROCESSING
# Pastikan fungsi ini sama saat training dan saat chatbot dipakai.
# =========================================================
def bersihkan_teks(teks):
    teks = str(teks).lower().strip()
    teks = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks)
    return teks.strip()


def baca_dataset(file):
    if file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file, encoding="utf-8-sig")

    wajib = {"pertanyaan", "jawaban", "intent"}
    hilang = wajib - set(df.columns)
    if hilang:
        raise ValueError(
            "Kolom wajib yang tidak ditemukan: " + ", ".join(sorted(hilang))
        )

    for kolom in ["pertanyaan", "jawaban", "intent"]:
        df[kolom] = df[kolom].fillna("").astype(str).str.strip()

    df = df[
        (df["pertanyaan"] != "")
        & (df["jawaban"] != "")
        & (df["intent"] != "")
    ].copy()

    df["teks_bersih"] = df["pertanyaan"].apply(bersihkan_teks)
    df = df[df["teks_bersih"] != ""]
    df = df.drop_duplicates(subset=["pertanyaan", "intent"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset tidak memiliki baris yang valid.")
    if df["intent"].nunique() < 2:
        raise ValueError("Dataset harus memiliki minimal dua intent.")

    return df


def buat_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("naive_bayes", MultinomialNB()),
    ])


def pilih_jawaban(pertanyaan, intent, dataset, model):
    """
    Pilih jawaban dari baris dataset dengan intent yang diprediksi.
    Di antara baris dalam intent itu, pilih pertanyaan yang paling mirip
    secara cosine similarity menggunakan TF-IDF milik model.
    """
    kandidat = dataset[dataset["intent"] == intent].copy()

    if kandidat.empty:
        return (
            "Maaf, saya belum memiliki jawaban yang sesuai. "
            "Silakan hubungi layanan akademik Fakultas Hukum UNISBA."
        ), 0.0, None

    vectorizer = model.named_steps["tfidf"]
    query_vec = vectorizer.transform([bersihkan_teks(pertanyaan)])
    kandidat_vec = vectorizer.transform(kandidat["teks_bersih"].tolist())

    skor = cosine_similarity(query_vec, kandidat_vec).flatten()
    indeks_terbaik = int(skor.argmax())
    baris_terbaik = kandidat.iloc[indeks_terbaik]

    return (
        str(baris_terbaik["jawaban"]),
        float(skor[indeks_terbaik]),
        str(baris_terbaik["pertanyaan"]),
    )


def proses_pertanyaan(pertanyaan, model, dataset, ambang_probabilitas):
    teks = bersihkan_teks(pertanyaan)

    if not teks:
        return {
            "jawaban": "Silakan tuliskan pertanyaan terlebih dahulu.",
            "intent": None,
            "probabilitas": 0.0,
            "kemiripan": 0.0,
            "pertanyaan_rujukan": None,
            "ditolak": True,
        }

    probabilitas = model.predict_proba([teks])[0]
    kelas = model.classes_
    indeks = int(probabilitas.argmax())
    intent = str(kelas[indeks])
    nilai_prob = float(probabilitas[indeks])

    # Probabilitas Naive Bayes bukan jaminan kebenaran.
    # Ambang ini adalah pengaman sederhana, bukan confidence terkalibrasi.
    if nilai_prob < ambang_probabilitas:
        return {
            "jawaban": (
                "Maaf, saya belum yakin memahami pertanyaan tersebut. "
                "Silakan gunakan kata kunci yang lebih spesifik, atau "
                "hubungi layanan akademik Fakultas Hukum UNISBA."
            ),
            "intent": intent,
            "probabilitas": nilai_prob,
            "kemiripan": 0.0,
            "pertanyaan_rujukan": None,
            "ditolak": True,
        }

    jawaban, kemiripan, pertanyaan_rujukan = pilih_jawaban(
        pertanyaan, intent, dataset, model
    )

    return {
        "jawaban": jawaban,
        "intent": intent,
        "probabilitas": nilai_prob,
        "kemiripan": kemiripan,
        "pertanyaan_rujukan": pertanyaan_rujukan,
        "ditolak": False,
    }


# =========================================================
# SIDEBAR: DATASET DAN PELATIHAN
# =========================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    file_dataset = st.file_uploader(
        "Unggah dataset (.xlsx atau .csv)",
        type=["xlsx", "csv"],
        help="Kolom wajib: pertanyaan, jawaban, intent.",
    )

    rasio_uji = st.slider(
        "Persentase data uji",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    ambang_probabilitas = st.slider(
        "Ambang probabilitas intent",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help=(
            "Jika probabilitas prediksi di bawah ambang ini, chatbot "
            "akan meminta mahasiswa memperjelas pertanyaan. Nilai ini "
            "perlu diuji pada data nyata."
        ),
    )

    tombol_latih = st.button(
        "🚀 Latih / Muat Dataset",
        type="primary",
        use_container_width=True,
    )


if tombol_latih:
    if file_dataset is None:
        st.sidebar.error("Unggah dataset terlebih dahulu.")
    else:
        try:
            dataset = baca_dataset(file_dataset)
            jumlah_per_intent = dataset["intent"].value_counts()

            if (jumlah_per_intent < 2).any():
                intent_kurang = jumlah_per_intent[jumlah_per_intent < 2].index.tolist()
                st.sidebar.error(
                    "Setiap intent harus memiliki minimal dua pertanyaan. "
                    "Periksa intent: " + ", ".join(intent_kurang)
                )
            else:
                data_latih, data_uji = train_test_split(
                    dataset,
                    test_size=rasio_uji / 100,
                    random_state=42,
                    stratify=dataset["intent"],
                )

                model = buat_model()
                model.fit(data_latih["teks_bersih"], data_latih["intent"])

                prediksi = model.predict(data_uji["teks_bersih"])
                akurasi = accuracy_score(data_uji["intent"], prediksi)

                st.session_state["model"] = model
                # Gunakan seluruh dataset untuk pencarian jawaban setelah
                # model dievaluasi pada data uji yang ditahan.
                st.session_state["dataset"] = dataset
                st.session_state["data_uji"] = data_uji.copy()
                st.session_state["prediksi_uji"] = prediksi
                st.session_state["akurasi"] = akurasi
                st.session_state["laporan"] = classification_report(
                    data_uji["intent"],
                    prediksi,
                    output_dict=True,
                    zero_division=0,
                )
                st.session_state["riwayat"] = []
                st.sidebar.success("Dataset berhasil dimuat dan model dilatih.")

        except Exception as e:
            st.sidebar.error(f"Gagal memproses dataset: {e}")


# =========================================================
# STATUS MODEL DAN EVALUASI
# =========================================================
if "model" not in st.session_state:
    st.warning(
        "Mulai dengan mengunggah dataset di panel kiri, lalu klik "
        "**Latih / Muat Dataset**. Dataset harus memiliki kolom "
        "`pertanyaan`, `jawaban`, dan `intent`."
    )
    st.stop()

model = st.session_state["model"]
dataset = st.session_state["dataset"]

tab_chat, tab_evaluasi, tab_unduh = st.tabs(
    ["💬 Chatbot", "📊 Evaluasi Model", "💾 Unduh Model"]
)


# =========================================================
# TAB CHATBOT
# =========================================================
with tab_chat:
    st.subheader("Tanyakan informasi akademik")

    st.caption(
        "Jawaban dipilih dari dataset berdasarkan intent hasil prediksi, "
        "kemudian dicocokkan dengan pertanyaan dataset yang paling mirip."
    )

    if "riwayat" not in st.session_state:
        st.session_state["riwayat"] = []

    for pesan in st.session_state["riwayat"]:
        with st.chat_message(pesan["role"]):
            st.markdown(pesan["content"])
            if pesan.get("detail"):
                with st.expander("Detail prediksi (untuk pengujian)"):
                    st.write(pesan["detail"])

    pertanyaan_user = st.chat_input("Contoh: Bagaimana cara mengisi KRS?")

    if pertanyaan_user:
        st.session_state["riwayat"].append({
            "role": "user",
            "content": pertanyaan_user,
        })

        hasil = proses_pertanyaan(
            pertanyaan_user,
            model,
            dataset,
            ambang_probabilitas,
        )

        st.session_state["riwayat"].append({
            "role": "assistant",
            "content": hasil["jawaban"],
            "detail": {
                "Intent prediksi": hasil["intent"],
                "Probabilitas Naive Bayes": round(hasil["probabilitas"], 4),
                "Skor kemiripan pertanyaan": round(hasil["kemiripan"], 4),
                "Pertanyaan dataset terpilih": hasil["pertanyaan_rujukan"],
                "Status": "Perlu klarifikasi" if hasil["ditolak"] else "Dijawab dari dataset",
            },
        })

        st.rerun()

    col_clear, col_info = st.columns([1, 3])
    with col_clear:
        if st.button("Hapus percakapan"):
            st.session_state["riwayat"] = []
            st.rerun()
    with col_info:
        st.caption(
            "Untuk penggunaan resmi, pastikan isi jawaban sudah diverifikasi "
            "terhadap pedoman akademik dan aturan yang berlaku saat ini."
        )


# =========================================================
# TAB EVALUASI
# =========================================================
with tab_evaluasi:
    st.subheader("Hasil pengujian klasifikasi intent")

    st.metric(
        "Accuracy",
        f"{st.session_state['akurasi'] * 100:.2f}%",
    )

    laporan = pd.DataFrame(st.session_state["laporan"]).transpose()
    st.dataframe(laporan.round(3), use_container_width=True)

    data_uji = st.session_state["data_uji"].copy()
    prediksi = st.session_state["prediksi_uji"]

    hasil_uji = data_uji[["pertanyaan", "intent"]].copy()
    hasil_uji["prediksi_intent"] = prediksi
    hasil_uji["benar"] = hasil_uji["intent"] == hasil_uji["prediksi_intent"]

    st.subheader("Prediksi pada data uji")
    st.dataframe(hasil_uji, use_container_width=True)

    st.caption(
        "Jika dataset yang diunggah sejak awal berisi parafrasa yang sangat "
        "mirip, pembagian acak masih dapat menghasilkan evaluasi yang terlalu "
        "optimistis. Untuk penelitian, sebaiknya simpan ID pertanyaan induk "
        "dan lakukan pembagian berdasarkan kelompok pertanyaan induk."
    )


# =========================================================
# TAB UNDUH MODEL
# =========================================================
with tab_unduh:
    st.subheader("Unduh model dan dataset")

    st.write(
        "Paket ini berisi pipeline TF-IDF + Naive Bayes dan dataset yang "
        "diperlukan untuk memilih jawaban. File `.pkl` hanya boleh dimuat "
        "dari sumber tepercaya."
    )

    paket = {
        "model": model,
        "dataset": dataset.drop(columns=["teks_bersih"], errors="ignore"),
        "format_versi": 1,
    }

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zip_file:
        zip_file.writestr(
            "chatbot_akademik.pkl",
            pickle.dumps(paket),
        )
        zip_file.writestr(
            "dataset_chatbot.csv",
            dataset.drop(columns=["teks_bersih"], errors="ignore").to_csv(
                index=False,
                encoding="utf-8-sig",
            ),
        )

    st.download_button(
        "⬇️ Unduh model + dataset (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="chatbot_akademik_fh_unisba.zip",
        mime="application/zip",
    )

    st.download_button(
        "⬇️ Unduh dataset CSV",
        data=dataset.drop(columns=["teks_bersih"], errors="ignore").to_csv(
            index=False,
            encoding="utf-8-sig",
        ),
        file_name="dataset_chatbot_akademik.csv",
        mime="text/csv",
    )
