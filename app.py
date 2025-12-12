import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Sistem Pakar Akademik Lengkap", layout="wide")

# --- 1. Fungsi Load & Train Model (FIXED) ---
@st.cache_resource
def load_and_train_model(use_grades=True):
    with st.spinner('Sedang memproses dataset & melatih AI...'):
        # 1. Download & Load Data
        path = kagglehub.dataset_download("larsen0966/student-performance-data-set")
        csv_path = None
        for root, dirs, files in os.walk(path):
            if "student-por.csv" in files:
                csv_path = os.path.join(root, "student-por.csv"); break
            elif "student-mat.csv" in files:
                csv_path = os.path.join(root, "student-mat.csv"); break
        
        try:
            df = pd.read_csv(csv_path)
            if len(df.columns) <= 1: df = pd.read_csv(csv_path, sep=';')
        except: return None, None, None, None, None, None

        # 2. Filter Kolom (Tanpa 'school')
        all_columns = [
            'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
            'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 
            'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
            'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
            'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3'
        ]
        available_cols = [col for col in all_columns if col in df.columns]
        
        # LOGIKA TOGGLE: Hapus G1 & G2 jika user tidak ingin menggunakannya
        if not use_grades:
            if 'G1' in available_cols: available_cols.remove('G1')
            if 'G2' in available_cols: available_cols.remove('G2')
            
        df = df[available_cols].copy()

        # 3. Preprocessing & Encoding
        X = df.drop('G3', axis=1)
        y = df['G3']
        
        # --- PENTING: SIMPAN URUTAN KOLOM ASLI ---
        # Ini untuk memastikan urutan saat prediksi sama persis dengan saat training
        train_columns = X.columns.tolist()

        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le

        # 4. Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        # 5. Hitung Feature Importance (Untuk Grafik)
        feature_importance = pd.DataFrame({
            'Fitur': X.columns,
            'Pentingnya': model.feature_importances_
        }).sort_values(by='Pentingnya', ascending=False)
        
        # Kita return 'train_columns' juga sekarang
        return model, df, score, encoders, feature_importance, train_columns

# --- 2. Logika Saran (Expert System) ---
def generate_advice(prediction, inputs):
    advice = []
    
    # Analisis Nilai
    if prediction < 10:
        advice.append("⚠️ **PERINGATAN:** Prediksi nilai akhir BAHAYA (Gagal).")
    elif prediction < 14:
        advice.append("ℹ️ **INFO:** Nilai Lulus, tapi perlu peningkatan.")
    else:
        advice.append("✅ **BAGUS:** Potensi nilai tinggi.")

    # Analisis Penyebab (Root Cause)
    if inputs.get('failures', 0) > 0:
        advice.append("🔄 **Remedial:** Sejarah kegagalan berpengaruh besar. Perlu bimbingan khusus.")
    if inputs.get('studytime', 2) < 2:
        advice.append("📚 **Belajar:** Tambah jam belajar minimal 1 jam per hari.")
    if inputs.get('absences', 0) > 10:
        advice.append("attendance **Kehadiran:** Jangan bolos! Absensi tinggi merusak nilai.")
    if (inputs.get('Dalc', 0) + inputs.get('Walc', 0)) > 5:
        advice.append("🍷 **Lifestyle:** Kurangi alkohol/nongkrong malam.")
    if inputs.get('Medu', 0) < 2 and inputs.get('Fedu', 0) < 2:
        advice.append("🏠 **Dukungan:** Orang tua mungkin perlu dilibatkan lebih aktif dalam memotivasi belajar.")

    return advice

# --- 3. UI Utama (LENGKAP) ---
def main():
    st.title("🎓 Sistem Analisis Performa Siswa")
    
    # --- Sidebar: Kontrol Model ---
    st.sidebar.header("⚙️ Pengaturan AI")
    use_grades = st.sidebar.checkbox("Sertakan Nilai Lalu (G1 & G2)?", value=True, 
                                     help="Centang: Akurasi tinggi (tapi bias nilai).\nHapus Centang: Analisis murni faktor sosial/ekonomi.")
    
    # Load Model (Menangkap return tambahan: train_columns)
    model, df_raw, score, encoders, importance, train_columns = load_and_train_model(use_grades)

    if model is not None:
        # Tampilkan Status Model
        st.sidebar.divider()
        st.sidebar.metric("Akurasi AI (R²)", f"{score:.2f}")
        
        # Visualisasi Faktor Penting
        st.subheader("📊 Faktor Penentu Nilai (Menurut AI)")
        st.caption("Semakin panjang bar, semakin berpengaruh faktor tersebut terhadap nilai akhir.")
        st.bar_chart(importance.set_index('Fitur').head(10)) 

        st.divider()
        st.subheader("📝 Input Data Siswa")
        
        # --- INPUT FORM (TABBED UI) ---
        user_input = {}
        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Keluarga", "👤 Personal", "🕒 Gaya Hidup", "📈 Akademik"])

        # Tab 1: Keluarga
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                user_input['address'] = st.selectbox("Alamat", df_raw['address'].unique(), format_func=lambda x: "Kota" if x == 'U' else "Desa")
                user_input['famsize'] = st.selectbox("Jml Keluarga", df_raw['famsize'].unique(), format_func=lambda x: ">3 Org" if x == 'GT3' else "≤3 Org")
                user_input['Pstatus'] = st.selectbox("Status Ortu", df_raw['Pstatus'].unique(), format_func=lambda x: "Bersama" if x == 'T' else "Pisah")
                user_input['guardian'] = st.selectbox("Wali", df_raw['guardian'].unique())
            with col2:
                user_input['Medu'] = st.slider("Pendidikan Ibu", 0, 4, 2, help="0: SD, 4: Sarjana")
                user_input['Fedu'] = st.slider("Pendidikan Ayah", 0, 4, 2)
                user_input['Mjob'] = st.selectbox("Pekerjaan Ibu", df_raw['Mjob'].unique())
                user_input['Fjob'] = st.selectbox("Pekerjaan Ayah", df_raw['Fjob'].unique())

        # Tab 2: Personal
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                user_input['sex'] = st.radio("Gender", df_raw['sex'].unique())
                user_input['age'] = st.number_input("Umur", 15, 22, 17)
                user_input['reason'] = st.selectbox("Alasan Sekolah", df_raw['reason'].unique())
                user_input['famrel'] = st.slider("Hubungan Keluarga (1-5)", 1, 5, 4)
            with col2:
                c1, c2 = st.columns(2)
                with c1:
                    user_input['schoolsup'] = st.selectbox("Bimbel Sekolah?", df_raw['schoolsup'].unique())
                    user_input['famsup'] = st.selectbox("Les Privat?", df_raw['famsup'].unique())
                    user_input['paid'] = st.selectbox("Les Berbayar?", df_raw['paid'].unique())
                    user_input['activities'] = st.selectbox("Ekskul?", df_raw['activities'].unique())
                with c2:
                    user_input['nursery'] = st.selectbox("Pernah PAUD?", df_raw['nursery'].unique())
                    user_input['higher'] = st.selectbox("Ingin Kuliah?", df_raw['higher'].unique())
                    user_input['internet'] = st.selectbox("Ada Internet?", df_raw['internet'].unique())
                    user_input['romantic'] = st.selectbox("Pacaran?", df_raw['romantic'].unique())

        # Tab 3: Gaya Hidup
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                user_input['traveltime'] = st.selectbox("Waktu Perjalanan", [1,2,3,4], format_func=lambda x: ["<15 mnt", "15-30 mnt", "30-60 mnt", ">60 mnt"][x-1])
                user_input['studytime'] = st.selectbox("Jam Belajar/Minggu", [1,2,3,4], format_func=lambda x: ["<2 Jam", "2-5 Jam", "5-10 Jam", ">10 Jam"][x-1])
                user_input['failures'] = st.selectbox("Jml Gagal Sebelumnya", [0, 1, 2, 3])
                user_input['absences'] = st.number_input("Jml Bolos", 0, 93, 2)
            with col2:
                user_input['freetime'] = st.slider("Waktu Luang", 1, 5, 3)
                user_input['goout'] = st.slider("Sering Keluar", 1, 5, 3)
                user_input['Dalc'] = st.slider("Alkohol (Weekday)", 1, 5, 1)
                user_input['Walc'] = st.slider("Alkohol (Weekend)", 1, 5, 1)
                user_input['health'] = st.slider("Kesehatan", 1, 5, 5)

        # Tab 4: Nilai (Dinamis)
        with tab4:
            if use_grades:
                st.info("Masukkan nilai periode sebelumnya untuk prediksi akurat.")
                c1, c2 = st.columns(2)
                user_input['G1'] = c1.number_input("Nilai Periode 1 (G1)", 0, 20, 12)
                user_input['G2'] = c2.number_input("Nilai Periode 2 (G2)", 0, 20, 12)
            else:
                st.warning("Mode 'Tanpa Nilai' aktif. Prediksi hanya berdasarkan faktor sosial & demografi.")

        # --- TOMBOL PREDIKSI ---
        if st.button("🚀 Analisis & Prediksi", type="primary"):
            
            # Persiapan Data
            input_df = pd.DataFrame([user_input])
            
            # PERBAIKAN UTAMA DISINI:
            # Gunakan 'train_columns' untuk memaksa urutan kolom sama persis dengan saat training
            input_df = input_df[train_columns]

            # Encoding Input User
            for col, encoder in encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col])
                    except:
                        pass 

            # Prediksi
            prediction = model.predict(input_df)[0]
            advice_list = generate_advice(prediction, user_input)

            # --- HASIL ---
            st.divider()
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.markdown("### Hasil Prediksi (G3)")
                st.metric("Nilai Akhir", f"{prediction:.2f} / 20")
                st.progress(min(prediction/20, 1.0))
                
                if prediction >= 15:
                    st.success("Sangat Baik (A)")
                elif prediction >= 10:
                    st.warning("Lulus (B/C)")
                else:
                    st.error("Gagal (D/E)")

            with col_res2:
                st.markdown("### 💡 Rekomendasi Perbaikan")
                if not advice_list:
                    st.write("Tidak ada isu kritikal yang ditemukan.")
                for item in advice_list:
                    st.write(item)

if __name__ == "__main__":
    main()