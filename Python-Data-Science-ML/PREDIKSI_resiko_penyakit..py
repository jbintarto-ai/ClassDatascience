import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle

# =====================================================================
# 1. MEMBUAT DATASET DENGAN NILAI RANDOM
# =====================================================================

# Jumlah sampel data
n_samples = 100

# Membuat nilai random untuk setiap fitur sesuai limit yang diberikan
gl_values = np.random.randint(80, 401, n_samples) # Glukosa (limit 80-400)
bmi_values = np.random.uniform(26, 41, n_samples) # BMI (limit 26-40)
kol_values = np.random.randint(60, 301, n_samples) # Kolesterol (limit 60-300)
djm_values = np.random.randint(90, 201, n_samples) # Detak_Jantung_Maks (limit 90-200)
umur_values = np.random.randint(25, 71, n_samples) # Umur (limit 25-70)
target_values = np.random.randint(0, 2, n_samples) # Target (limit 0 dan 1)

data_random = {
    'Glukosa': gl_values,
    'BMI': bmi_values,
    'Kolesterol': kol_values,
    'Detak_Jantung_Maks': djm_values,
    'Umur': umur_values,
    'Target': target_values
}

df_random = pd.DataFrame(data_random)

print("DataFrame dengan nilai random berhasil dibuat:")
print(df_random)

# Pisahkan Fitur (X) dan Target/Label (y)
X_new = df_random[['Glukosa', 'BMI', 'Kolesterol', 'Detak_Jantung_Maks', 'Umur']]
y_new = df_random['Target']

# Bagi data menjadi 80% Training dan 20% Testing
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42, stratify=y_new)

# =====================================================================
# 2. MEMBUAT PIPELINE MACHINE LEARNING UNTUK PREDIKSI RISIKO PENYAKIT
# =====================================================================
risk_prediction_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # Mengisi nilai NaN dengan median
    ('scaler', StandardScaler()), # Menyamakan skala data
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) # Algoritma Klasifikasi
])

# Melatih pipeline dengan data baru
risk_prediction_pipeline.fit(X_train_new, y_train_new)

# =====================================================================
# 3. MENYIMPAN MODEL PIPELINE
# =====================================================================
model_filename_new = 'prediksi-Resiko-penyakit.pkl'
with open(model_filename_new, 'wb') as file:
    pickle.dump(risk_prediction_pipeline, file)

print(f"\nModel pipeline berhasil disimpan sebagai '{model_filename_new}'")

import pickle
from sklearn.metrics import accuracy_score, classification_report

# =====================================================================
# 4. EVALUASI PERFORMA MODEL PIPELINE 'prediksi-Resiko-penyakit.pkl'
# =====================================================================

# Muat model pipeline yang telah disimpan
model_filename_new = 'prediksi-Resiko-penyakit.pkl'
try:
    with open(model_filename_new, 'rb') as file:
        loaded_risk_pipeline = pickle.load(file)
    print(f"Model pipeline '{model_filename_new}' berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: File model '{model_filename_new}' tidak ditemukan. Pastikan sudah dibuat dan disimpan.")
    loaded_risk_pipeline = None

if loaded_risk_pipeline:
    # Lakukan prediksi pada data uji (X_test_new dan y_test_new harus tersedia dari sel sebelumnya)
    y_pred_new = loaded_risk_pipeline.predict(X_test_new)

    print("\n=== EVALUASI PERFORMA MODEL 'prediksi-Resiko-penyakit.pkl' ===")
    print(f"Akurasi Model: {accuracy_score(y_test_new, y_pred_new) * 100:.2f}%")
    print("\nClassification Report:")
    # Menentukan nama kelas untuk laporan klasifikasi
    # Asumsi: 0 = 'Tidak Berisiko', 1 = 'Berisiko'
    print(classification_report(y_test_new, y_pred_new, target_names=['Tidak Berisiko', 'Berisiko'], zero_division=0))
else:
    print("Evaluasi performa model tidak dapat dilakukan karena model gagal dimuat.")
