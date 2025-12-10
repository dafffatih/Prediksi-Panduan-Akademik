import streamlit as st
import pandas as pd
import numpy as np
import kagglehub # Library baru
import os # Untuk mengatur path file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Sistem Pakar Akademik", layout="wide")

# --- 1. Fungsi Load & Train Model (Updated with KaggleHub) ---
# --- 1. Fungsi Load & Train Model (Versi Perbaikan) ---
# --- 1. Fungsi Load & Train Model (FIX SEPARATOR) ---
@st.cache_resource
def load_and_train_model():
    with st.spinner('Sedang mendownload dan mencari dataset...'):
        try:
            # 1. Download dataset
            path = kagglehub.dataset_download("larsen0966/student-performance-data-set")
            
            # 2. Cari file csv
            csv_path = None
            for root, dirs, files in os.walk(path):
                # Kita prioritaskan student-por.csv karena student-mat.csv tadi hilang
                if "student-por.csv" in files:
                    csv_path = os.path.join(root, "student-por.csv")
                    break
                # Cadangan jika student-mat muncul lagi
                elif "student-mat.csv" in files:
                    csv_path = os.path.join(root, "student-mat.csv")
                    break
            
            if csv_path is None:
                st.error(f"File CSV tidak ditemukan di folder: {path}")
                return None, None, None

            # 3. Baca CSV (PERBAIKAN DISINI)
            # Hapus sep=';' agar pandas mendeteksi otomatis (biasanya koma)
            try:
                df = pd.read_csv(csv_path)
                
                # Cek darurat: jika kolomnya ternyata masih salah (misal cuma 1 kolom)
                # Berarti dia memang butuh titik koma
                if len(df.columns) <= 1:
                     df = pd.read_csv(csv_path, sep=';')
            except:
                 df = pd.read_csv(csv_path, sep=';')
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            return None, None, None

    # --- Preprocessing ---
    # Kita pastikan kolom yang kita panggil benar-benar ada di dataframe
    # Kadang nama kolom di kaggle beda dikit (misal huruf besar/kecil)
    expected_cols = ['sex', 'age', 'studytime', 'failures', 'schoolsup', 
                     'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    
    # Validasi kolom tersedia
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Kolom berikut hilang dari dataset: {missing_cols}")
        st.write("Kolom yang tersedia:", df.columns.tolist())
        return None, None, None

    selected_features = ['sex', 'age', 'studytime', 'failures', 'schoolsup', 
                         'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    
    data = df[selected_features].copy()
    target = df['G3'] 

    le = LabelEncoder()
    data['sex'] = le.fit_transform(data['sex']) 
    data['schoolsup'] = le.fit_transform(data['schoolsup'])

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    
    return model, df, score

# --- 2. Logika Sistem Pakar (Rule-Based System) ---
def generate_advice(prediction, studytime, failures, absences, alcohol_consumption):
    advice = []
    
    # Aturan 1: Prediksi Nilai
    if prediction < 10:
        advice.append("⚠️ **PERINGATAN:** Prediksi nilai akhir di bawah standar kelulusan (KKM). Perlu belajar ekstra keras.")
    elif prediction < 14:
        advice.append("ℹ️ **INFO:** Anda di jalur aman, tapi tingkatkan lagi untuk nilai A.")
    else:
        advice.append("✅ **BAGUS:** Pertahankan kinerja! Potensi nilai A.")

    # Aturan 2: Waktu Belajar
    if studytime == 1:
        advice.append("📚 **Tips Belajar:** Waktu belajar < 2 jam/minggu terlalu sedikit. Coba tambah 30 menit setiap hari.")
    
    # Aturan 3: Absensi
    if absences > 10:
        advice.append("attendance **Kehadiran:** Absensi Anda tinggi. Ini faktor utama penurunan nilai. Jangan bolos lagi.")

    # Aturan 4: Kegagalan
    if failures > 0:
        advice.append("🔄 **Remedial:** Karena ada riwayat gagal sebelumnya, disarankan ambil kelas tambahan.")

    # Aturan 5: Alkohol/Hiburan Malam
    if alcohol_consumption > 5:
        advice.append("🍷 **Kesehatan:** Kurangi nongkrong/minum di akhir pekan. Kesehatan fisik berpengaruh ke otak.")

    return advice

# --- 3. UI Utama ---
def main():
    st.title("🎓 Prediksi & Panduan Akademik (Kaggle Integrated)")
    st.markdown("Dataset diambil otomatis dari: *larsen0966/student-performance-data-set*")
    
    # Load Model
    model, df, score = load_and_train_model()

    if model is not None:
        st.sidebar.success(f"Model Ready! Akurasi: {score:.2f}")
        
        st.subheader("Simulasi Data Siswa")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Akademik")
            g1 = st.number_input("Nilai G1 (Periode 1)", 0, 20, 12)
            g2 = st.number_input("Nilai G2 (Periode 2)", 0, 20, 12)
            failures = st.selectbox("Pernah Gagal Pelajaran?", [0, 1, 2, 3])
            schoolsup = st.selectbox("Ikut Bimbel Sekolah?", ["Tidak", "Ya"])

        with col2:
            st.markdown("### 🕒 Kebiasaan")
            studytime = st.selectbox("Waktu Belajar Mingguan", [1, 2, 3, 4], 
                                     format_func=lambda x: ["<2 Jam", "2-5 Jam", "5-10 Jam", ">10 Jam"][x-1])
            absences = st.slider("Jumlah Bolos/Absen", 0, 93, 2)
            goout = st.slider("Sering Keluar Main? (1-5)", 1, 5, 3)

        with col3:
            st.markdown("### 👤 Personal")
            sex = st.radio("Gender", ["Pria", "Wanita"])
            age = st.slider("Umur", 15, 22, 17)
            freetime = st.slider("Waktu Luang (1-5)", 1, 5, 3)
            dalc = st.slider("Alkohol (Hari Kerja)", 1, 5, 1)
            walc = st.slider("Alkohol (Weekend)", 1, 5, 1)
            health = st.slider("Kesehatan (1-5)", 1, 5, 5)

        if st.button("🚀 Prediksi Hasil Belajar"):
            # Encoding input
            sex_enc = 1 if sex == "Pria" else 0
            sup_enc = 1 if schoolsup == "Ya" else 0
            
            # Array Input
            input_data = np.array([[sex_enc, age, studytime, failures, sup_enc, 
                                    freetime, goout, dalc, walc, health, absences, g1, g2]])
            
            # Prediksi
            prediction = model.predict(input_data)[0]
            advice_list = generate_advice(prediction, studytime, failures, absences, (dalc + walc))

            st.divider()
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric("Prediksi Nilai Akhir (G3)", f"{prediction:.2f} / 20")
                if prediction >= 10:
                    st.success("LULUS")
                else:
                    st.error("GAGAL")
            
            with c2:
                st.info("### 💡 Rekomendasi Sistem Pakar")
                for tips in advice_list:
                    st.write(tips)

if __name__ == "__main__":
    main()