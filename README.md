# 🧠 Burnout Classification Using SBERT, TF-IDF, BoW, LDA & BiLSTM

**A Comprehensive NLP Pipeline for Burnout Detection (3-Class Classification)**

Repository ini berisi implementasi *end-to-end* untuk **klasifikasi burnout** menjadi tiga kategori:

* **College Burnout**
* **Worker Burnout**
* **No Burnout**

Menggunakan kombinasi teknik **classic machine learning** dan **deep learning**, termasuk:

* Sentence-BERT
* TF-IDF (Unigram & N-gram)
* Bag-of-Words
* LDA Topic Modeling
* Naive Bayes, SVM, XGBoost, Logistic Regression, Random Forest
* Bidirectional LSTM


## 📌 1. Deskripsi Singkat Proyek

Proyek ini bertujuan membangun sistem klasifikasi burnout berbasis teks dari data media sosial. Dua pendekatan utama digunakan:

### **A. Machine Learning Klasik**

Dengan fitur:

* TF-IDF Unigram
* TF-IDF N-gram
* Bag-of-Words
* LDA Topic Distribution

Model yang digunakan:

* Multinomial Naive Bayes
* Random Forest
* SVM
* Logistic Regression
* XGBoost

### **B. Deep Learning**

Model:

* **Bidirectional LSTM** dengan trainable embedding

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

## ⚙️ 3. Fitur Utama Pipeline

### ✔ Automated Labeling menggunakan SBERT

* Menghitung similarity teks–query
* Mengambil label berdasarkan skor tertinggi
* Menggunakan threshold untuk validitas label

### ✔ Text Preprocessing Lengkap

* Lowercasing
* Menghapus URL, emoji, angka, simbol
* Stopword removal
* Normalisasi spasi

### ✔ Oversampling Kelas Minoritas

Menggunakan **RandomOverSampler** untuk menyeimbangkan dataset.

### ✔ Empat Metode Feature Extraction

| Metode         | Deskripsi                              |
| -------------- | -------------------------------------- |
| TF-IDF Unigram | Representasi klasik berbasis frekuensi |
| TF-IDF N-gram  | Menangkap konteks lebih panjang        |
| BoW Unigram    | Baseline sederhana                     |
| LDA Features   | Distribusi probabilistik topik         |

### ✔ Bidirectional LSTM

* Embedding 128 dim
* LSTM dua arah (64 unit)
* Dense 64 neurons
* Dropout

---

## 📊 4. Hasil Evaluasi

### **A. Machine Learning**

| Model               | Feature        | Akurasi |
| ------------------- | -------------- | ------- |
| Naive Bayes         | Bag of Words   | 84%     |
| Naive Bayes         | TF-IDF Unigram | 84%     |
| Naive Bayes         | TF-IDF N-grams | 86%     |
| Naive Bayes         | LDA            | 45%     |
| XGBoost             | Bag of Words   | 85%     |
| XGBoost             | TF-IDF Unigram | 86%     |
| XGBoost             | TF-IDF N-grams | 87%     |
| XGBoost             | LDA            | 81%     |
| Logistic Regression | Bag of Words   | 90%     |
| Logistic Regression | TF-IDF Unigram | 89%     |
| Logistic Regression | TF-IDF N-grams | 91%     |
| Logistic Regression | LDA            | 45%     |
| SVM                 | Bag of Words   | 91%     |
| SVM                 | TF-IDF Unigram | 91%     |
| SVM                 | TF-IDF N-grams | 94%     |
| SVM                 | LDA            | 45%     |

### **B. Deep Learning**

| Model                  | Akurasi |
| ---------------------- | ------- |
| **Bidirectional LSTM** | **93%** |

---

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

