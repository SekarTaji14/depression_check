import streamlit as st
import pandas as pd
import joblib

# =========================
# Load artefak final
# =========================
preprocess = joblib.load("preprocess_final.joblib")
lr_model = joblib.load("lr_final.joblib")
lr_meta = joblib.load("lr_meta.joblib")

selected_features = lr_meta["selected_features"]
best_threshold = float(lr_meta["best_threshold"])

# Feature names encoded (harus sama persis seperti encoding kamu)
num_cols = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Work/Study Hours", "Financial Stress"]
ord_cols = ["Sleep Duration", "Dietary Habits", "Degree"]
binary_cols = ["Gender", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]
all_feature_names = num_cols + ord_cols + binary_cols

# Mapping manual (harus sama dengan training)
gender_map = {"Female": 0, "Male": 1}
suicidal_map = {"No": 0, "Yes": 1}
family_map = {"No": 0, "Yes": 1}

degree_map = {
    "Class 12": "Pre-University",

    "BA": "Undergraduate", "BSc": "Undergraduate", "BCA": "Undergraduate", "BBA": "Undergraduate",
    "B.Com": "Undergraduate", "B.Tech": "Undergraduate", "BE": "Undergraduate", "B.Pharm": "Undergraduate",
    "B.Ed": "Undergraduate", "B.Arch": "Undergraduate", "BHM": "Undergraduate",

    "MA": "Postgraduate", "MSc": "Postgraduate", "MBA": "Postgraduate", "M.Tech": "Postgraduate",
    "ME": "Postgraduate", "MCA": "Postgraduate", "M.Com": "Postgraduate", "M.Pharm": "Postgraduate",
    "M.Ed": "Postgraduate", "MHM": "Postgraduate",

    "MBBS": "Professional",
    "LLB": "Law", "LLM": "Law",
    "PhD": "Doctoral", "MD": "Doctoral",
    "Others": "Others"
}

sleep_code_to_label = {
    0: "Less than 5 hours",
    1: "5-6 hours",
    2: "7-8 hours",
    3: "More than 8 hours"
}

def apply_manual_mapping(df_in: pd.DataFrame) -> pd.DataFrame:
    df2 = df_in.copy()
    df2["Gender"] = df2["Gender"].map(gender_map)
    df2["Have you ever had suicidal thoughts ?"] = df2["Have you ever had suicidal thoughts ?"].map(suicidal_map)
    df2["Family History of Mental Illness"] = df2["Family History of Mental Illness"].map(family_map)
    df2["Degree"] = df2["Degree"].map(degree_map).fillna("Others")
    return df2


# =========================
# UI setup + style
# =========================
st.set_page_config(page_title="Cek Kesehatan Mental", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.hero {font-size: 38px; font-weight: 900; line-height: 1.1; margin-bottom: 6px;}
.sub {opacity: 0.88; font-size: 15px;}
.badge {
  display:inline-block; padding: 6px 12px; border-radius: 999px;
  border: 1px solid rgba(49,51,63,.25); font-size: 12px; opacity: .92;
  margin-right: 6px; margin-top: 6px;
}
.card {
  padding: 18px; border-radius: 16px;
  border: 1px solid rgba(49,51,63,.2); background: rgba(250,250,250,.55);
}
.small {font-size: 13px; opacity: .85;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero">Cek Kesehatan Mental Kamu Yuk 🧠✨</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Isi data singkat di bawah ini untuk melihat <b>prediksi risiko depresi</b>. '
    'Aplikasi ini dibangun menggunakan <span class="badge">Machine Learning</span> '
    'dengan Algoritma <span class="badge">Logistic Regression</span>.</div>',
    unsafe_allow_html=True
)
st.write("")

# =========================
# TENTANG MODEL (TIDAK MUNCUL DULU)
# =========================
with st.expander("🧾 Tentang Model"):
    st.markdown(f"""
**Algoritma:** Logistic Regression  
**Threshold final:** {best_threshold:.2f}  
**Jumlah fitur dipakai:** {len(selected_features)}  

Aplikasi ini demo implementasi model hasil penelitian (tuning + threshold).  
**Hasil prediksi bukan diagnosis klinis.**
""")
    if st.button("🔄 Reset Form", key="reset_btn"):
        st.rerun()

# =========================
# FORM INPUT
# =========================
with st.form("input_form"):
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📌 Data Akademik & Finansial")
        st.caption("Isi sesuai kondisi kamu **dalam beberapa minggu terakhir**.")

        st.markdown("**Umur**")
        st.caption("Rentang umur yang didukung model: **18–35 tahun**.")
        age = st.number_input("Umur (tahun)", min_value=18, max_value=35, value=20, step=1, key="age")

        st.markdown("**Academic Pressure (Tekanan Akademik)**")
        st.caption(
            "Skala 1–5:\n"
            "• 1 = Hampir tidak ada tekanan\n"
            "• 2 = Tekanan ringan\n"
            "• 3 = Tekanan sedang\n"
            "• 4 = Tekanan tinggi\n"
            "• 5 = Tekanan sangat tinggi"
        )
        academic_pressure = st.radio("Pilih tingkat tekanan akademik", [1,2,3,4,5], horizontal=True, key="ap")

        st.markdown("**Study Satisfaction (Kepuasan Belajar)**")
        st.caption(
            "Skala 1–5:\n"
            "• 1 = Sangat tidak puas\n"
            "• 2 = Tidak puas\n"
            "• 3 = Cukup puas\n"
            "• 4 = Puas\n"
            "• 5 = Sangat puas"
        )
        study_satisfaction = st.radio("Pilih tingkat kepuasan belajar", [1,2,3,4,5], horizontal=True, key="ss")

        st.markdown("**Financial Stress (Stres Finansial)**")
        st.caption(
            "Skala 1–5:\n"
            "• 1 = Tidak merasa stres finansial\n"
            "• 2 = Stres ringan\n"
            "• 3 = Stres sedang\n"
            "• 4 = Stres tinggi\n"
            "• 5 = Stres sangat tinggi"
        )
        financial_stress = st.radio("Pilih tingkat stres finansial", [1,2,3,4,5], horizontal=True, key="fs")

        st.markdown("**CGPA / IPK**")
        st.caption("Masukkan IPK terakhir (0–10).")
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1, key="cgpa")

        st.markdown("**Work / Study Hours**")
        st.caption("Total jam belajar dan/atau kerja per hari (0–24 jam).")
        work_hours = st.slider("Jam per hari", 0, 24, 6, key="hours")

    with col2:
        st.markdown("### 🧩 Gaya Hidup & Riwayat")
        st.caption("Bagian ini berkaitan dengan kebiasaan dan faktor risiko pribadi.")

        st.markdown("**Gender**")
        gender = st.radio("Pilih gender", ["Female", "Male"], horizontal=True, key="gender")

        st.markdown("**Sleep Duration (Durasi Tidur)**")
        st.caption(
            "Gunakan kode berikut:\n"
            "• 0 = Kurang dari 5 jam\n"
            "• 1 = 5–6 jam\n"
            "• 2 = 7–8 jam\n"
            "• 3 = Lebih dari 8 jam"
        )
        sleep_code = st.radio("Pilih kode durasi tidur", [0,1,2,3], horizontal=True, key="sleep")
        st.caption(f"Pilihan kamu: **{sleep_code} → {sleep_code_to_label[int(sleep_code)]}**")

        st.markdown("**Dietary Habits (Pola Makan)**")
        st.caption(
            "• Unhealthy = sering makan tidak teratur / junk food\n"
            "• Moderate = cukup seimbang\n"
            "• Healthy = teratur dan bergizi"
        )
        diet = st.selectbox("Pilih pola makan", ["Unhealthy", "Moderate", "Healthy"], key="diet")

        st.markdown("**Degree (Jenjang Pendidikan)**")
        st.caption("Pilih jenjang pendidikan saat ini. (Akan di-group otomatis sesuai penelitian.)")
        degree = st.selectbox("Degree", [
            "Class 12","BA","BSc","BCA","BBA","B.Com","B.Tech","BE","B.Pharm","B.Ed","B.Arch","BHM",
            "MA","MSc","MBA","M.Tech","ME","MCA","M.Com","M.Pharm","M.Ed","MHM",
            "MBBS","LLB","LLM","PhD","MD","Others"
        ], key="degree")

        st.markdown("**Suicidal Thoughts**")
        st.caption("Apakah kamu pernah memiliki pikiran untuk menyakiti diri sendiri?")
        suicidal = st.radio("Jawaban suicidal thoughts", ["No", "Yes"], horizontal=True, key="suicidal")

        st.markdown("**Family History of Mental Illness**")
        st.caption("Apakah ada anggota keluarga dengan riwayat gangguan mental?")
        family = st.radio("Jawaban family history", ["No", "Yes"], horizontal=True, key="family")

    st.markdown("---")
    st.markdown("### 🔍 Sudah yakin dengan jawabanmu?")
    submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

# =========================
# Prediction
# =========================
if submitted:
    sleep_label = sleep_code_to_label[int(sleep_code)]

    input_raw = pd.DataFrame([{
        "Age": int(age),
        "Academic Pressure": int(academic_pressure),
        "CGPA": float(cgpa),
        "Study Satisfaction": int(study_satisfaction),
        "Work/Study Hours": int(work_hours),
        "Financial Stress": int(financial_stress),
        "Gender": gender,
        "Have you ever had suicidal thoughts ?": suicidal,
        "Family History of Mental Illness": family,
        "Sleep Duration": sleep_label,
        "Dietary Habits": diet,
        "Degree": degree
    }])

    input_mapped = apply_manual_mapping(input_raw)

    X_enc = preprocess.transform(input_mapped)
    X_enc_df = pd.DataFrame(X_enc, columns=all_feature_names)

    X_final = X_enc_df[selected_features]
    proba = float(lr_model.predict_proba(X_final)[0][1])
    pred = 1 if proba >= best_threshold else 0

    st.write("")
    st.markdown("## 🧾 Hasil Prediksi")

    if pred == 1:
        st.markdown(f"""
        <div class="card">
            <div class="badge">Kategori</div>
            <h2 style="margin:10px 0 4px 0;">😔 Risiko Tinggi</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card">
            <div class="badge">Kategori</div>
            <h2 style="margin:10px 0 4px 0;">🙂 Risiko Rendah</h2>
        </div>
        """, unsafe_allow_html=True)

    st.progress(min(max(proba, 0.0), 1.0))
    st.caption("Progress bar menunjukkan probabilitas (0–1).")

    with st.expander("🔍 Lihat data input (untuk dokumentasi)"):
        st.dataframe(input_raw, use_container_width=True)

    st.info("Jika kamu merasa tidak baik-baik saja, pertimbangkan menghubungi layanan konseling kampus/tenaga profesional.")

