# Python-Data-Science
Kelas Python Machine learning session 1

## Unsupervised Learning
nsupervised Learning (Pembelajaran Tanpa Pengawasan) adalah salah satu jenis machine learning di mana algoritma dilatih menggunakan dataset yang tidak memiliki label (unlabeled data).

Artinya, data yang dimasukkan tidak memiliki jawaban benar atau salah yang mendampinginya. Tugas algoritma adalah menemukan pola, struktur tersembunyi, atau hubungan antar-data itu sendiri secara mandiri.

## Kategori Utama Unsupervised Learning
Secara umum, algoritma ini dibagi menjadi dua tugas utama:

1. Clustering (Pengelompokan)
Tujuannya adalah membagi data menjadi beberapa kelompok (cluster) berdasarkan kemiripan karakteristiknya. Data dalam satu kelompok memiliki kemiripan yang tinggi, sedangkan data antar-kelompok sangat berbeda.

2. Association (Asosiasi)
Tujuannya adalah menemukan aturan dan hubungan menarik antara variabel-variabel dalam dataset yang besar. Ini mencari tahu kecenderungan "jika membeli barang A, kemungkinan besar juga akan membeli barang B".

# Model Algoritma Populer 
Berikut adalah beberapa algoritma Unsupervised Learning yang paling sering digunakan beserta contoh penerapannya di dunia nyata:
1. K-Means Clustering (Clustering)Algoritma yang membagi data menjadi sejumlah $K$ kelompok. Algoritma ini bekerja dengan cara menentukan titik pusat cluster (centroid) secara acak, lalu memasukkan data ke centroid terdekat, dan memperbarui posisi centroid tersebut berulang kali hingga stabil.
2. Hierarchical Clustering (Clustering)Berbeda dengan K-Means yang langsung membagi data secara instan, algoritma ini membangun hierarki kelompok berbentuk seperti struktur pohon (disebut Dendrogram). Prosesnya bisa dari bawah ke atas (menggabungkan data kecil menjadi besar) atau dari atas ke bawah (memecah data besar menjadi kecil).

3. Apriori Algorithm (Association)Algoritma klasik yang digunakan untuk menemukan hubungan atau kombinasi produk yang sering dibeli bersamaan oleh konsumen dalam sebuah transaksi.

4. Principal Component Analysis / PCA (Dimensionality Reduction)Meski fokusnya adalah pengurangan dimensi (mereduksi jumlah fitur/kolom yang terlalu banyak), PCA termasuk dalam unsupervised learning karena bekerja tanpa label. PCA menyederhanakan data yang kompleks dengan tetap mempertahankan informasi penting sebanyak mungkin.
