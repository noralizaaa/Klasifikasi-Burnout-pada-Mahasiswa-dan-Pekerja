# 🧠 Burnout Classification Using Classic ML, Deep Learning & Transformers

**A Comprehensive NLP Pipeline for Burnout Detection (3-Class Classification)**

Repository ini berisi implementasi *end-to-end* untuk **klasifikasi burnout** menjadi tiga kategori:

* **College Burnout**
* **Worker Burnout**
* **No Burnout**

Menggunakan kombinasi teknik **Classic Machine Learning**, **Deep Learning**, dan **Transfer Learning**, termasuk:

* Sentence-BERT (untuk Auto-Labeling)
* TF-IDF (Unigram & N-gram) & Bag-of-Words
* LDA Topic Modeling
* Naive Bayes, SVM, XGBoost, Logistic Regression, Random Forest
* Bidirectional LSTM
* **DistilBERT (Pre-trained Transformer)**


## 📌 1. Deskripsi Singkat Proyek

Proyek ini bertujuan membangun sistem klasifikasi burnout berbasis teks dari data media sosial. Tiga pendekatan utama digunakan:

### **A. Machine Learning Klasik**
Model tradisional dengan *feature engineering* manual:
* **Fitur:** TF-IDF (Unigram/N-gram), Bag-of-Words, LDA Topic Distribution.
* **Model:** Multinomial Naive Bayes, Random Forest, SVM, Logistic Regression, XGBoost.

### **B. Deep Learning**
Pendekatan berbasis jaringan saraf tiruan:
* **Model:** **Bidirectional LSTM** dengan *trainable embedding* layer.

### **C. Pre-trained Transformer (Transfer Learning)**
Pendekatan mutakhir menggunakan model bahasa yang telah dilatih sebelumnya:
* **Model:** **DistilBERT** (`distilbert-base-uncased`) yang di-*fine-tune* khusus untuk dataset burnout ini.

---

## 🗂 2. Struktur Repository

Dataset diberi label otomatis menggunakan **Sentence-BERT** melalui kemiripan semantik dengan query setiap kategori.

Pipeline mencakup:

* Data loading
* Semantic labeling (SBERT)
* Thresholding
* Oversampling
* Text preprocessing
* Feature extraction
* Model training
* Evaluation & confusion matrix

---

## 🗂 2. Struktur Repository

```
📁 burnout-classification/
│
├── README.md
├── Burnout_3_Class_Filtered.csv      # Dataset final hasil filtering & labeling
└── Klasifikasi Burnout pada Mahasiswa dan Pekerja.ipynb   # Notebook utama (Google Colab)
```

---

---

## ⚙️ 3. Fitur Utama Pipeline

### ✔ Automated Labeling menggunakan SBERT
* Menghitung similarity teks–query menggunakan `all-mpnet-base-v2`.
* Mengambil label berdasarkan skor tertinggi dengan threshold validitas.

### ✔ Text Preprocessing Lengkap
* Lowercasing, Hapus URL/Emoji/Simbol, Stopword removal, Normalisasi.

### ✔ Oversampling Kelas Minoritas
Menggunakan **RandomOverSampler** untuk menyeimbangkan dataset sebelum training.

### ✔ Feature Extraction & Tokenization
* **Classic ML:** TF-IDF, BoW, LDA.
* **Deep Learning:** Tokenizer & Sequencing (Keras).
* **Transformer:** DistilBertTokenizer (Hugging Face).

---

## 📊 4. Hasil Evaluasi

Berikut adalah perbandingan performa antar pendekatan:

### **A. Machine Learning Klasik (Baseline)**

| Model               | Feature        | Akurasi |
| ------------------- | -------------- | ------- |
| Naive Bayes         | TF-IDF N-grams | 86%     |
| XGBoost             | TF-IDF N-grams | 87%     |
| Logistic Regression | TF-IDF N-grams | 91%     |
| SVM                 | TF-IDF N-grams | 94%     |
| **Random Forest** | **TF-IDF** | **96%** |

### **B. Deep Learning (BiLSTM)**

| Model               | Akurasi |
| ------------------- | ------- |
| Bidirectional LSTM  | 93%     |

### **C. Pre-trained Transformer (DistilBERT)**

Model DistilBERT menunjukkan performa yang sangat stabil dan tinggi pada semua kelas.

| Kelas            | Precision | Recall | F1-Score | Support |
| ---------------- | :-------: | :----: | :------: | :-----: |
| college_burnout  | 0.94      | 0.98   | 0.96     | 3329    |
| no_burnout       | 0.97      | 1.00   | 0.99     | 3330    |
| worker_burnout   | 0.98      | 0.91   | 0.94     | 3330    |
| **Accuracy** |           |        | **0.96** | **9989**|

---

## 🏁 5. Kesimpulan

1.  **Top Performance:** **DistilBERT** dan **Random Forest** berbagi posisi teratas dengan akurasi **96%**.
2.  **Keunggulan DistilBERT:** Mencapai **F1-Score 0.99** pada kelas *No Burnout* dan sangat presisi membedakan topik burnout pekerja vs mahasiswa.
3.  **Efektivitas Deep Learning:** BiLSTM (93%) performanya baik, namun masih sedikit di bawah DistilBERT yang memanfaatkan *knowledge transfer* dari corpus bahasa Inggris yang masif.
4.  **Rekomendasi:**
    * Gunakan **DistilBERT** jika resource komputasi (GPU) tersedia dan menginginkan pemahaman konteks semantik terbaik.
    * Gunakan **Random Forest + TF-IDF** untuk solusi yang lebih ringan, cepat, dan mudah di-deploy pada CPU.

---

## 🚀 6. Cara Menjalankan

### 1. Clone Repository
```bash
git clone [https://github.com/your-username/burnout-classification.git](https://github.com/your-username/burnout-classification.git)
cd burnout-classification
## 🏁 5. Kesimpulan

* **Random Forest + TF-IDF** memberikan akurasi **tertinggi (96%)**, mengungguli pendekatan deep learning.
* **Bidirectional LSTM** mencapai **93%**, tetap kompetitif dalam memahami konteks sekuensial.
* Perbedaan performa dipengaruhi oleh ukuran dataset, karakteristik teks pendek, dan efektivitas TF-IDF untuk klasifikasi berbasis kata.

**Kesimpulan utama:**
➡ *Untuk dataset kecil–menengah dengan teks pendek, metode klasik berbasis TF-IDF masih menjadi solusi paling efektif dan stabil.*

---

## 🚀 6. Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/your-username/burnout-classification.git
cd burnout-classification
```

### 2. Jalankan Notebook

Buka file:

```
Klasifikasi Burnout pada Mahasiswa dan Pekerja.ipynb
```

Direkomendasikan menggunakan **Google Colab** (opsional GPU untuk BiLSTM).

---

## 🔧 7. Teknologi yang Digunakan

* Python 3.10+
* Sentence-BERT (`all-mpnet-base-v2`)
* Scikit-learn
* XGBoost
* TensorFlow/Keras
* NLTK
* Pandas
* Matplotlib & Seaborn

---

## 🤝 8. Kontribusi

Kontribusi sangat diterima. Anda dapat menambahkan:

* Model transformer (fine-tuning BERT)
* Dashboard analisis
* Evaluasi tambahan (ROC, PR Curve, SHAP)

---

