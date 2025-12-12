import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Sistem Pakar Akademik (No School)", layout="wide")

# --- 1. Fungsi Load & Train Model (Removed 'school') ---
@st.cache_resource
def load_and_train_model():
    with st.spinner('Sedang mendownload dan memproses dataset...'):
        try:
            # 1. Download dataset
            path = kagglehub.dataset_download("larsen0966/student-performance-data-set")
            
            # 2. Cari file csv
            csv_path = None
            for root, dirs, files in os.walk(path):
                if "student-por.csv" in files:
                    csv_path = os.path.join(root, "student-por.csv")
                    break
                elif "student-mat.csv" in files:
                    csv_path = os.path.join(root, "student-mat.csv")
                    break
            
            if csv_path is None:
                st.error("File CSV tidak ditemukan.")
                return None, None, None, None

            # 3. Baca CSV
            try:
                df = pd.read_csv(csv_path)
                if len(df.columns) <= 1:
                     df = pd.read_csv(csv_path, sep=';')
            except:
                 df = pd.read_csv(csv_path, sep=';')
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            return None, None, None, None

    # --- Preprocessing ---
    # Daftar kolom (Kita SUDAH MENGHAPUS 'school' dari sini)
    all_columns = [
        'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
        'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 
        'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
        'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
        'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3'
    ]
    
    # Filter hanya kolom yang ada
    available_cols = [col for col in all_columns if col in df.columns]
    df = df[available_cols].copy()

    # Pisahkan Fitur dan Target
    target_col = 'G3'
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col]

    # --- Automatic Label Encoding ---
    encoders = {}
    
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    
    return model, df, score, encoders

# --- 2. Logika Sistem Pakar (Advice) ---
def generate_advice(prediction, inputs):
    advice = []
    
    if prediction < 10:
        advice.append("⚠️ **PERINGATAN KRITIS:** Prediksi nilai akhir di bawah KKM. Risiko tidak lulus tinggi.")
    elif prediction < 15:
        advice.append("ℹ️ **INFO:** Nilai cukup aman, namun perlu dorongan untuk mencapai predikat Sangat Baik.")
    else:
        advice.append("✅ **EXCELLENT:** Pertahankan ritme belajar ini!")

    if inputs['failures'] > 0:
        advice.append("🔄 **Sejarah Akademik:** Riwayat kegagalan masa lalu mempengaruhi prediksi. Ambil kelas remedial.")
    
    if inputs['studytime'] < 2:
        advice.append("📚 **Waktu Belajar:** Anda belajar kurang dari 2 jam/minggu. Tingkatkan minimal menjadi 5-10 jam.")
    
    if inputs['absences'] > 10:
        advice.append("attendance **Kehadiran:** Tingkat ketidakhadiran sangat tinggi. Ini faktor penentu negatif terbesar.")
        
    if inputs['Dalc'] + inputs['Walc'] > 5:
        advice.append("🍷 **Gaya Hidup:** Konsumsi alkohol cukup tinggi. Kurangi demi fokus dan kesehatan kognitif.")

    return advice

# --- 3. UI Utama ---
def main():
    st.title("🎓 Sistem Pakar Akademik (General)")
    st.markdown("Prediksi performa siswa tanpa bias sekolah asal.")
    
    model, df_raw, score, encoders = load_and_train_model()

    if model is not None:
        st.sidebar.success(f"Model Accuracy (R²): {score:.2f}")
        
        st.subheader("📋 Input Data Siswa")
        user_input = {}

        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Keluarga", "👤 Personal", "🕒 Gaya Hidup", "📊 Nilai"])

        with tab1: # School dihapus, mulai dari Address
            col1, col2 = st.columns(2)
            with col1:
                # 'school' removed here
                user_input['address'] = st.selectbox("Alamat", df_raw['address'].unique(), format_func=lambda x: "Perkotaan" if x == 'U' else "Pedesaan")
                user_input['famsize'] = st.selectbox("Ukuran Keluarga", df_raw['famsize'].unique(), format_func=lambda x: ">3 Orang" if x == 'GT3' else "≤3 Orang")
                user_input['Pstatus'] = st.selectbox("Status Ortu", df_raw['Pstatus'].unique(), format_func=lambda x: "Hidup Bersama" if x == 'T' else "Terpisah")
                user_input['guardian'] = st.selectbox("Wali Murid", df_raw['guardian'].unique())
            with col2:
                user_input['Medu'] = st.slider("Pendidikan Ibu (0-4)", 0, 4, 2)
                user_input['Fedu'] = st.slider("Pendidikan Ayah (0-4)", 0, 4, 2)
                user_input['Mjob'] = st.selectbox("Pekerjaan Ibu", df_raw['Mjob'].unique())
                user_input['Fjob'] = st.selectbox("Pekerjaan Ayah", df_raw['Fjob'].unique())

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                user_input['sex'] = st.radio("Gender", df_raw['sex'].unique())
                user_input['age'] = st.number_input("Umur", 15, 22, 17)
                user_input['reason'] = st.selectbox("Alasan Memilih Sekolah", df_raw['reason'].unique())
                user_input['famrel'] = st.slider("Hubungan Keluarga (1-5)", 1, 5, 4)
                
            with col2:
                c1, c2 = st.columns(2)
                with c1:
                    user_input['schoolsup'] = st.selectbox("Bimbel Sekolah?", df_raw['schoolsup'].unique())
                    user_input['famsup'] = st.selectbox("Bimbel Keluarga?", df_raw['famsup'].unique())
                    user_input['paid'] = st.selectbox("Les Berbayar?", df_raw['paid'].unique())
                    user_input['activities'] = st.selectbox("Ekskul?", df_raw['activities'].unique())
                with c2:
                    user_input['nursery'] = st.selectbox("Pernah TK?", df_raw['nursery'].unique())
                    user_input['higher'] = st.selectbox("Ingin Kuliah?", df_raw['higher'].unique())
                    user_input['internet'] = st.selectbox("Akses Internet?", df_raw['internet'].unique())
                    user_input['romantic'] = st.selectbox("Pacaran?", df_raw['romantic'].unique())

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                user_input['traveltime'] = st.selectbox("Waktu Perjalanan", [1,2,3,4], format_func=lambda x: ["<15 min", "15-30 min", "30-60 min", ">60 min"][x-1])
                user_input['studytime'] = st.selectbox("Waktu Belajar", [1,2,3,4], format_func=lambda x: ["<2 Jam", "2-5 Jam", "5-10 Jam", ">10 Jam"][x-1])
                user_input['failures'] = st.selectbox("Jumlah Kegagalan Lalu", [0, 1, 2, 3])
                user_input['absences'] = st.number_input("Jumlah Absen/Bolos", 0, 93, 2)
            with col2:
                user_input['freetime'] = st.slider("Waktu Luang (1-5)", 1, 5, 3)
                user_input['goout'] = st.slider("Sering Keluar (1-5)", 1, 5, 3)
                user_input['Dalc'] = st.slider("Alkohol Weekday (1-5)", 1, 5, 1)
                user_input['Walc'] = st.slider("Alkohol Weekend (1-5)", 1, 5, 1)
                user_input['health'] = st.slider("Kesehatan (1-5)", 1, 5, 5)

        with tab4:
            st.info("Masukkan nilai periode sebelumnya untuk prediksi nilai akhir (G3)")
            c1, c2 = st.columns(2)
            user_input['G1'] = c1.number_input("Nilai Periode 1 (G1)", 0, 20, 12)
            user_input['G2'] = c2.number_input("Nilai Periode 2 (G2)", 0, 20, 12)

        if st.button("🚀 Prediksi Hasil Belajar", type="primary"):
            input_df = pd.DataFrame([user_input])
            feature_cols = [c for c in df_raw.columns if c != 'G3']
            input_df = input_df[feature_cols]

            for col, encoder in encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col])
                    except:
                        pass # Handle unseen labels gracefully

            prediction = model.predict(input_df)[0]
            advice_list = generate_advice(prediction, user_input)

            st.divider()
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.markdown("### Hasil Prediksi")
                st.metric("Estimasi Nilai Akhir (G3)", f"{prediction:.2f} / 20")
                st.progress(min(prediction/20, 1.0))
                
                if prediction >= 15:
                    st.success("Sangat Baik (A)")
                elif prediction >= 10:
                    st.warning("Lulus (B/C)")
                else:
                    st.error("Gagal (D/E)")

            with col_res2:
                st.markdown("### 💡 Rekomendasi")
                for item in advice_list:
                    st.write(item)

if __name__ == "__main__":
    main()