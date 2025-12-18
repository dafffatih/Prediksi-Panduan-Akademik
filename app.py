import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Sistem Pakar Akademik AI", layout="wide", page_icon="🎓")

# --- 1. Fungsi Load & Train Model ---
@st.cache_resource  # 1. Caching: Prevents retraining the model on every page interaction (saves time/memory).
def load_and_train_model(use_grades=True):
    with st.spinner('Sedang memproses dataset & melatih AI...'):
        
        # --- 1. LOAD DATA ---
        # Downloads the Portuguese student dataset from Kaggle
        path = kagglehub.dataset_download("larsen0966/student-performance-data-set")
        csv_path = None
        for root, dirs, files in os.walk(path):
            if "student-por.csv" in files:
                csv_path = os.path.join(root, "student-por.csv"); break
        
        try:
            df = pd.read_csv(csv_path)
            # Check for separator: This dataset often uses ';' instead of ','
            if len(df.columns) <= 1: df = pd.read_csv(csv_path, sep=';')
        except: return None, None, None, None, None, None

        # Keep a "clean" copy for display in the UI (before we convert text to numbers)
        df_raw_display = df.copy()

        # --- 2. FILTER & PREPARE FEATURES ---
        all_columns = [
            'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
            'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 
            'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
            'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
            'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3'
        ]
        # Only keep columns that actually exist in the file
        available_cols = [col for col in all_columns if col in df.columns]
        
        # LOGIC FOR "PREDICTION MODE":
        # If use_grades is False, we drop G1 (1st period grade) and G2 (2nd period).
        # This allows the model to predict purely based on demographics/habits,
        # rather than "cheating" by looking at previous exam scores.
        if not use_grades:
            if 'G1' in available_cols: available_cols.remove('G1')
            if 'G2' in available_cols: available_cols.remove('G2')
            
        df = df[available_cols].copy()

        # --- 3. ENCODING (Text -> Numbers) ---
        X = df.drop('G3', axis=1) # Features (Input)
        y = df['G3']              # Target (Output we want to predict)
        train_columns = X.columns.tolist()

        encoders = {}
        for col in X.columns:
            # AI cannot read text like "yes"/"no" or "F"/"M".
            # LabelEncoder converts them to numbers (e.g., yes=1, no=0).
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le # Save encoder to translate user input later

        # --- 4. TRAINING ---
        # Split data: 80% for training, 20% for testing (to check accuracy)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Algorithm: RandomForest
        # Why? It uses multiple decision trees to average results, reducing errors 
        # and handling complex relationships better than simple linear regression.
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy (R^2 score) on data the model hasn't seen
        score = model.score(X_test, y_test)
        
        # --- 5. EXPLAINABILITY ---
        # Extracts which factors influenced the prediction the most (e.g., Absences, Study Time)
        feature_importance = pd.DataFrame({
            'Fitur': X.columns,
            'Pentingnya': model.feature_importances_
        }).sort_values(by='Pentingnya', ascending=False)
        
        return model, df_raw_display, score, encoders, feature_importance, train_columns

# --- 2. Logika Saran (Expert System) ---
def generate_advice(prediction, inputs):
    advice = []
    if prediction < 10:
        advice.append("⚠️ **PERINGATAN:** Prediksi nilai akhir BAHAYA (Gagal).")
    elif prediction < 14:
        advice.append("ℹ️ **INFO:** Nilai Lulus, tapi perlu peningkatan.")
    else:
        advice.append("✅ **BAGUS:** Potensi nilai tinggi.")

    if inputs.get('failures', 0) > 0:
        advice.append("🔄 **Remedial:** Sejarah kegagalan berpengaruh besar. Perlu bimbingan khusus.")
    if inputs.get('studytime', 2) < 2:
        advice.append("📚 **Belajar:** Tambah jam belajar minimal 1 jam per hari.")
    if inputs.get('absences', 0) > 10:
        advice.append("🚫 **Kehadiran:** Jangan bolos! Absensi tinggi sangat merusak nilai.")
    
    return advice

# --- 3. UI Utama ---
def main():
    # --- Sidebar Navbar ---
    st.sidebar.title("📌 Menu Utama")
    page = st.sidebar.radio("Pilih Halaman:", ["🔮 Prediksi Nilai", "📊 Lihat Dataset"])
    
    st.sidebar.divider()
    st.sidebar.header("⚙️ Konfigurasi AI")
    use_grades = st.sidebar.checkbox("Sertakan Nilai G1 & G2", value=True)

    # Load data dan model
    model, df_display, score, encoders, importance, train_columns = load_and_train_model(use_grades)

    if model is not None:
        # --- HALAMAN 1: PREDIKSI ---
        if page == "🔮 Prediksi Nilai":
            st.title("🎓 Sistem Analisis Performa Siswa")
            st.sidebar.metric("Akurasi Model (R²)", f"{score:.2f}")

            st.subheader("📊 Faktor Penentu Nilai (Feature Importance)")
            st.bar_chart(importance.set_index('Fitur').head(10)) 

            st.divider()
            st.subheader("📝 Masukkan Data Siswa")
            
            user_input = {}
            tab1, tab2, tab3, tab4 = st.tabs(["🏠 Keluarga", "👤 Personal", "🕒 Gaya Hidup", "📈 Akademik"])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    user_input['address'] = st.selectbox("Alamat", df_display['address'].unique(), format_func=lambda x: "Kota" if x == 'U' else "Desa")
                    user_input['famsize'] = st.selectbox("Jml Keluarga", df_display['famsize'].unique(), format_func=lambda x: ">3 Org" if x == 'GT3' else "≤3 Org")
                    user_input['Pstatus'] = st.selectbox("Status Ortu", df_display['Pstatus'].unique(), format_func=lambda x: "Bersama" if x == 'T' else "Pisah")
                with col2:
                    user_input['Medu'] = st.slider("Pendidikan Ibu (0-4)", 0, 4, 2)
                    user_input['Fedu'] = st.slider("Pendidikan Ayah (0-4)", 0, 4, 2)
                    user_input['Mjob'] = st.selectbox("Pekerjaan Ibu", df_display['Mjob'].unique())
                    user_input['Fjob'] = st.selectbox("Pekerjaan Ayah", df_display['Fjob'].unique())
                    user_input['guardian'] = st.selectbox("Wali", df_display['guardian'].unique())

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    user_input['sex'] = st.radio("Gender", df_display['sex'].unique())
                    user_input['age'] = st.number_input("Umur", 15, 22, 17)
                with col2:
                    user_input['reason'] = st.selectbox("Alasan Sekolah", df_display['reason'].unique())
                    user_input['famrel'] = st.slider("Kualitas Hubungan Keluarga", 1, 5, 4)

            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    user_input['studytime'] = st.selectbox("Jam Belajar", [1,2,3,4], format_func=lambda x: ["<2 Jam", "2-5 Jam", "5-10 Jam", ">10 Jam"][x-1])
                    user_input['traveltime'] = st.selectbox("Waktu Perjalanan", [1,2,3,4], format_func=lambda x: ["<15 mnt", "15-30 mnt", "30-60 mnt", ">60 mnt"][x-1])
                    user_input['absences'] = st.number_input("Jumlah Absensi (Bolos)", 0, 100, 0)
                with col2:
                    user_input['goout'] = st.slider("Sering Keluar Main", 1, 5, 2)
                    user_input['health'] = st.slider("Tingkat Kesehatan", 1, 5, 5)
                    user_input['Dalc'] = st.slider("Konsumsi Alkohol (Hari Kerja)", 1, 5, 1)
                    user_input['Walc'] = st.slider("Konsumsi Alkohol (Akhir Pekan)", 1, 5, 1)

            with tab4:
                # Kolom-kolom boolean lainnya
                c1, c2 = st.columns(2)
                with c1:
                    user_input['schoolsup'] = st.selectbox("Dukungan Sekolah?", df_display['schoolsup'].unique())
                    user_input['famsup'] = st.selectbox("Dukungan Keluarga?", df_display['famsup'].unique())
                    user_input['paid'] = st.selectbox("Kelas Tambahan Berbayar?", df_display['paid'].unique())
                    user_input['activities'] = st.selectbox("Kegiatan Ekskul?", df_display['activities'].unique())
                with c2:
                    user_input['nursery'] = st.selectbox("Pernah TK/PAUD?", df_display['nursery'].unique())
                    user_input['higher'] = st.selectbox("Ingin Kuliah?", df_display['higher'].unique())
                    user_input['internet'] = st.selectbox("Akses Internet?", df_display['internet'].unique())
                    user_input['romantic'] = st.selectbox("Sedang Berpacaran?", df_display['romantic'].unique())
                    user_input['failures'] = st.selectbox("Jumlah Kegagalan Kelas Lalu", [0, 1, 2, 3])

                if use_grades:
                    st.divider()
                    user_input['G1'] = st.number_input("Nilai Periode 1 (G1)", 0, 20, 12)
                    user_input['G2'] = st.number_input("Nilai Periode 2 (G2)", 0, 20, 12)

            if st.button("🚀 Jalankan Prediksi AI", type="primary"):
                input_df = pd.DataFrame([user_input])
                input_df = input_df[train_columns] # Pastikan urutan kolom pas

                for col, encoder in encoders.items():
                    if col in input_df.columns:
                        input_df[col] = encoder.transform(input_df[col])

                prediction = model.predict(input_df)[0]
                advice_list = generate_advice(prediction, user_input)

                st.divider()
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric("Prediksi Nilai Akhir (G3)", f"{prediction:.2f} / 20")
                    st.progress(min(prediction/20, 1.0))
                with res_col2:
                    st.markdown("### 💡 Saran Pakar")
                    for a in advice_list: st.write(a)

        # --- HALAMAN 2: DATASET VIEWER ---
        elif page == "📊 Lihat Dataset":
            st.title("📂 Eksplorasi Data Mentah")
            st.write("Data ini diambil langsung dari dataset Student Performance di Kaggle.")
            
            # Statistik Ringkas
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Baris", len(df_display))
            c2.metric("Total Kolom", len(df_display.columns))
            c3.metric("Rata-rata Nilai G3", round(df_display['G3'].mean(), 2))

            st.divider()
            
            # Tabel Interaktif
            st.subheader("📋 Tabel Data Lengkap")
            st.dataframe(df_display, use_container_width=True)

            # Fitur Download
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Dataset (CSV)", csv, "data_siswa.csv", "text/csv")

            # Visualisasi Tambahan Sederhana
            st.divider()
            st.subheader("📈 Distribusi Nilai Akhir (G3)")
            st.bar_chart(df_display['G3'].value_counts())

if __name__ == "__main__":
    main()