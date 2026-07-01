import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

# Mengatur seed agar data yang dihasilkan konsisten
np.random.seed(42)
# Jumlah sampel data
n_samples = 200
# Generate fitur kesehatan
data = {
    'Usia': np.random.randint(15, 80, n_samples),
    'BMI': np.round(np.random.uniform(17.0, 35.0, n_samples), 1),
    'Tekanan_Darah': np.random.randint(90, 180, n_samples),
    'Gula_Darah': np.random.randint(70, 250, n_samples),
    'Aktivitas_Fisik': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], n_samples)
}
df = pd.DataFrame(data)
# Simulasi Logika Diagnosa (Target)
# Seseorang berisiko (1) jika Gula Darah > 140 atau BMI > 30, dengan sedikit variasi acak
conditions = (
    (df['Gula_Darah'] > 140) |
    (df['BMI'] > 30) |
    ((df['Tekanan_Darah'] > 140) & (df['Usia'] > 50))
)
df['Diagnosa'] = conditions.astype(int)
# Menambahkan sedikit 'noise' agar model ML belajar lebih keras
noise = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
df['Diagnosa'] = np.where(noise, 1 - df['Diagnosa'], df['Diagnosa'])
# Simpan ke CSV
df.to_csv('data_kesehatan.csv', index=False)
print("File 'data_kesehatan.csv' berhasil dibuat!")
# 1. Load Data
df = pd.read_csv('data_kesehatan.csv')
X = df.drop('Diagnosa', axis=1)
y = df['Diagnosa']
# 2. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. Definisi Preprocessing
numeric_features = ['Usia', 'BMI', 'Tekanan_Darah', 'Gula_Darah']
categorical_features = ['Aktivitas_Fisik']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
# 4. Membangun Pipeline Lengkap
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
# 5. Training
model_pipeline.fit(X_train, y_train)
# Dengan menyimpan pipeline (Sekarang setelah model terlatih)
joblib.dump(model_pipeline, 'model_diagnosa_kesehatan.pkl')