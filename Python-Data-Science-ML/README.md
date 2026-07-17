# Python-Data-Science
Kelas Python Machine learning session 1
1. ML Pipeline
2. ML supervised linierRegression 


# apa itu Machine learning ( Pembelajaran mesin)
Machine learning ( Pembelajaran mesin) adalah cabang dari Kecerdasan Buatan yang berfokus pada pengembangan model dan algoritma yang memungkinkan komputer belajar dari data tanpa diprogram secara eksplisit untuk setiap tugas

<img width="774" height="625" alt="ML" src="https://github.com/user-attachments/assets/37d8bfaf-5bba-45d2-b066-3bd11fd6f6ae" />

## Pembelajaran Mesin pada dasarnya terbagi menjadi tiga jenis utama:
1. Supervised Learning(Pembelajaran Terawasi ) : Melatih model pada data berlabel untuk memprediksi atau mengklasifikasikan data baru yang belum pernah dilihat sebelumnya.
2. Unsupervised Learning ( Pembelajaran Tanpa Pengawasan ) : Menemukan pola atau kelompok dalam data yang tidak berlabel, seperti pengelompokan atau pengurangan dimensi.
3. Reinforcement Learning (Pembelajaran Penguatan) : Belajar melalui coba-coba untuk memaksimalkan imbalan, ideal untuk tugas pengambilan keputusan.

## Machine Learning Pipeline
Pipeline Machine Learning (ML Pipeline) adalah rangkaian alur kerja otomatis yang menggabungkan seluruh tahapan pengembangan model kecerdasan buatan, mulai dari pemrosesan data mentah hingga model siap digunakan di lingkungan produksi. Sistem ini berfungsi untuk menyederhanakan, menstandarisasi, dan mempercepat siklus hidup pengembangan kecerdasan buatan
### Komponen Utama:
1. Data Cleaning: Menangani nilai yang hilang (missing values), misalnya mengisi rata-rata pada kolom tekanan darah.
2. Feature Scaling: Menyamakan skala data (misalnya, umur vs. kadar kolesterol) menggunakan StandardScaler agar model tidak bias.
3. Encoding: Mengubah data kategori (seperti jenis kelamin atau riwayat merokok) menjadi angka.
4. Model Estimator: Algoritma klasifikasi seperti Random Forest atau Support Vector Machine (SVM).

# (supervised learning)
Pembelajaran terawasi (supervised learning) adalah jenis pembelajaran mesin di mana sebuah model belajar dari data berlabel, artinya setiap input memiliki output yang benar. Model tersebut membandingkan prediksinya dengan hasil aktual dan terus meningkat seiring waktu untuk meningkatkan akurasi.

##  Fitur utamanya adalah:
Data Berlabel : Setiap input memiliki output yang diketahui.
Belajar dari Kesalahan : Menyesuaikan diri untuk mengurangi kesalahan prediksi
Tujuan : Membuat prediksi akurat berdasarkan data baru.
## Cara Kerja supervised learning
Cara kerja pembelajaran mesin terawasi mengikuti langkah-langkah utama berikut:
1. Kumpulkan Data Berlabel
Kumpulkan dataset di mana setiap input memiliki output (label) yang benar dan diketahui.
Contoh : Gambar angka tulisan tangan dengan angka aslinya sebagai label.
2. Pisahkan Dataset
Bagilah data menjadi data pelatihan (sekitar 80%) dan data pengujian (sekitar 20%).
Model tersebut akan belajar dari data pelatihan dan dievaluasi pada data pengujian.
3. Melatih Model
Masukkan data pelatihan (input dan labelnya) ke algoritma pembelajaran terawasi yang sesuai (seperti Pohon Keputusan, SVM, atau Regresi Linier).
Model ini mencoba menemukan pola yang memetakan input ke output yang benar.
4. Validasi dan Uji Model
Evaluasi model menggunakan data pengujian yang belum pernah dilihat sebelumnya.
Model tersebut memprediksi output dan prediksi ini dibandingkan dengan label sebenarnya untuk menghitung akurasi atau kesalahan.
5. Menerapkan dan Memprediksi pada Data Baru
Setelah model berkinerja baik, model tersebut dapat digunakan untuk memprediksi output untuk data yang benar-benar baru dan belum pernah dilihat sebelumnya.
## Jenis-jenis supervised learning
* Classification: Di ​​mana outputnya berupa variabel kategorikal (misalnya, email spam vs. bukan spam, ya vs. tidak). 
* Regression: Di ​​mana outputnya adalah variabel kontinu (misalnya, memprediksi harga rumah, harga saham). 
## Algoritma Pembelajaran Mesin Terawasi
Pembelajaran terawasi dapat dibagi lagi menjadi beberapa jenis yang berbeda, masing-masing dengan karakteristik dan aplikasinya yang unik. Berikut adalah jenis algoritma pembelajaran terawasi yang paling umum:
Regresi Linier : Regresi linier adalah jenis algoritma regresi pembelajaran terawasi yang digunakan untuk memprediksi nilai keluaran kontinu. Ini adalah salah satu algoritma paling sederhana dan paling banyak digunakan dalam pembelajaran terawasi.  


# Cara  Install Lewat requirements.txt
pip install -r requirements.txt
