# Python Data Science - Neural Network

Repository ini berisi materi singkat dan contoh sederhana tentang neural network menggunakan Python.

## Apa itu Neural Network?
Neural network adalah model machine learning yang terinspirasi dari cara kerja neuron di otak manusia. Model ini mampu mempelajari pola dari data dan digunakan untuk klasifikasi, regresi, pengenalan gambar, dan NLP.

## Komponen Utama
- Input layer: menerima data
- Hidden layer: memproses data
- Output layer: menghasilkan prediksi
- Weight dan bias: parameter yang dipelajari
- Activation function: ReLU, sigmoid, tanh

## Cara Kerja
1. Data masuk ke input layer.
2. Setiap neuron menghitung weighted sum.
3. Hasil diproses dengan activation function.
4. Output dibandingkan dengan target.
5. Error dikirim balik melalui backpropagation untuk memperbarui weight.


## Contoh Sederhana
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
```

## instalasi requirements.txt
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## solusi codespace sinkro repo 
1. Cek Status Git
```
cd /workspaces/Python-Data-Science
git status
```
2. Cek Remote Repository
```
git remote -v
```
3. Cek Branch Saat Ini
```
git branch -a
```
### Kemungkinan Masalah & Solusi:
A. Branch tidak tersinkronisasi:
```
git fetch origin
git pull origin NeuralNetwork
```
B. Ada perubahan lokal yang belum di-commit:
```
git add .
git commit -m "Update materi neural network dan requirements"
git push origin NeuralNetwork
```
C. Jika ada konflik dengan default branch (Main):
```
git fetch origin
git merge origin/Main
# Resolve conflicts jika ada
git push origin NeuralNetwork
```
D. Reset ke remote state (jika ingin mulai bersih):
```
git fetch origin
git reset --hard origin/NeuralNetwork
```
4. Verifikasi Autentikasi
Pastikan GitHub credentials sudah ter-setup:
```
git config --list | grep user
gh auth status
```
Jika belum login, gunakan:
```
gh auth login
```
menjalankan perintah 
```
git status  
git remote -v
```
