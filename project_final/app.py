from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Muat model dan pengolah data
model = joblib.load('medical_model.pkl')
tfidf = joblib.load('medical_tfidf.pkl')
params = joblib.load('scaling_params.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose_web', methods=['POST'])
def diagnose_web():
    # Ambil data dari form HTML
    keluhan = request.form.get('keluhan')
    suhu = float(request.form.get('suhu'))
    tensi = float(request.form.get('tensi'))

    # Preprocessing & Prediksi
    text_vector = tfidf.transform([keluhan.lower()]).toarray()
    suhu_norm = (suhu - params['min']) / (params['max'] - params['min'])
    numeric_features = np.array([[suhu_norm, tensi]])
    features = np.hstack((text_vector, numeric_features))

    # Tambahkan kamus ini di bawah variabel model
    terapi_map = {
    'Flu/Infeksi Saluran Napas': 'Istirahat total, minum air hangat, dan konsumsi vitamin C.',
    'Jantung': 'Segera periksa ke dokter spesialis jantung, hindari aktivitas berat.',
    'Hipotensi': 'Konsumsi makanan mengandung garam dan minum air putih yang cukup.',
    'Apendisitis': 'Segera ke UGD untuk pemeriksaan bedah, jangan menunda.',
    'Alergi Kulit':'Hindari pemicu alergi, gunakan krim antihistamin, dan konsultasikan ke dokter kulit.'
    }

    prediction = model.predict(features)[0]
    rekomendasis = terapi_map.get(prediction, 'Konsultasikan gejala Anda ke dokter umum.')
    # Kirim hasil kembali ke halaman index.html
     

if __name__ == '__main__':
   if __name__ == '__main__':
    app.run(port=5000, debug=True) # Tambahkan debug=True
