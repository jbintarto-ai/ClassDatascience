import pandas as pd
import pickle

print("=== FORM INPUT COMMAND LINE: PREDIKSI RISIKO PENYAKIT ===")

# --- Load the trained model pipeline once ---
model_filename = 'prediksi-Resiko-penyakit.pkl'
try:
    with open(model_filename, 'rb') as file:
        loaded_risk_pipeline = pickle.load(file)
    print(f"Model pipeline '{model_filename}' berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: File model '{model_filename}' tidak ditemukan. Pastikan sudah dibuat dan disimpan.")
    loaded_risk_pipeline = None
except Exception as e:
    print(f"Error saat memuat model: {e}")
    loaded_risk_pipeline = None

if loaded_risk_pipeline:
    try:
        # Fungsi untuk mendapatkan input dengan validasi
        def get_validated_input(prompt, min_val, max_val, input_type):
            while True:
                try:
                    value = input(f"{prompt} ({min_val}-{max_val}): ")
                    if input_type == int:
                        value = int(value)
                    else: # float
                        value = float(value)
                    if min_val <= value <= max_val:
                        return value
                    else:
                        print(f"Input harus dalam rentang {min_val}-{max_val}.")
                except ValueError:
                    print("Input tidak valid. Harap masukkan angka.")

        # Ambil input dari pengguna
        gl_val = get_validated_input("Masukkan nilai Glukosa", 80, 400, int)
        bmi_val = get_validated_input("Masukkan nilai BMI", 26.0, 40.0, float)
        kol_val = get_validated_input("Masukkan nilai Kolesterol", 60, 300, int)
        djm_val = get_validated_input("Masukkan nilai Detak Jantung Maks", 90, 200, int)
        umur_val = get_validated_input("Masukkan nilai Umur", 25, 70, int)

        # Buat DataFrame untuk pasien baru
        pasien_baru_data_cli = {
            'Glukosa': [gl_val],
            'BMI': [bmi_val],
            'Kolesterol': [kol_val],
            'Detak_Jantung_Maks': [djm_val],
            'Umur': [umur_val]
        }
        pasien_baru_df_cli = pd.DataFrame(pasien_baru_data_cli)

        # Lakukan prediksi
        prediksi_risiko_cli = loaded_risk_pipeline.predict(pasien_baru_df_cli)
        probabilitas_risiko_cli = loaded_risk_pipeline.predict_proba(pasien_baru_df_cli)

        print("\n=== HASIL PREDIKSI RESIKO PENYAKIT ===")
        if prediksi_risiko_cli[0] == 1:
            print(f"Hasil: Berisiko Tinggi Terkena Penyakit diabetes dan jantung")
            print(f"Tingkat Probabilitas Resiko: {probabilitas_risiko_cli[0][1] * 100:.1f}%")
        else:
            print(f"Hasil: Tidak Berisiko Tinggi Terkena Penyakit")
            print(f"Tingkat Probabilitas Resiko: {probabilitas_risiko_cli[0][1] * 100:.1f}% (Rendah)")
    except Exception as e:
        print(f"Terjadi kesalahan saat prediksi: {e}")
else:
    print("Prediksi tidak dapat dilakukan karena model gagal dimuat.")
