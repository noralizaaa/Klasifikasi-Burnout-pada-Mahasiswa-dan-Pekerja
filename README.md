---

# 🧠 Burnout Classification Using SBERT, TF-IDF, BoW, LDA & BiLSTM

**A Comprehensive NLP Pipeline for Burnout Detection (3-Class Classification)**

Repository ini berisi implementasi lengkap *end-to-end pipeline* untuk melakukan **klasifikasi burnout** menjadi 3 kategori:

* **College Burnout**
* **Worker Burnout**
* **No Burnout**

Menggunakan kombinasi teknik **classic machine learning** dan **deep learning**, termasuk:

* Sentence-BERT for semantic similarity
* TF-IDF (Unigram & N-gram)
* Bag-of-Words
* LDA Topic Features
* Naive Bayes
* SVM
* XGBoost
* Logistic Regression
* Random Forest
* Bidirectional LSTM

---

## 📌 **1. Deskripsi Singkat Proyek**

Proyek ini bertujuan membangun sistem klasifikasi burnout berbasis teks dari dataset sosial media. Klasifikasi dilakukan melalui 2 pendekatan:

### **A. Klasik (Machine Learning)**

Meliputi:

* TF-IDF Unigrams
* TF-IDF N-grams
* Bag-of-Words (BoW)
* LDA Topic Distribution

Model klasik yang digunakan:

* **Multinomial Naive Bayes**
* **Random Forest**
* **SVM**
* **Logistic Regression**
* **XGBoost**

### **B. Deep Learning**

Model yang digunakan:

* **Bidirectional LSTM** dengan embedding trainable

Dataset awal diekstraksi dari Hugging Face dan kemudian diberi label otomatis menggunakan **Sentence-BERT** berdasarkan kemiripan semantik dengan query setiap kategori.

Pipeline ini mencakup:

* Data loading
* Semantic labeling (SBERT)
* Threshold filtering
* Oversampling
* Preprocessing teks lengkap
* Feature extraction (4 metode)
* Model training & evaluation
* Confusion matrix visualization

---

## 🗂 **2. Struktur Repository**

```
📁 burnout-classification/
│
├── README.md
├── Burnout_3_Class_Filtered.csv     # Dataset final setelah filtering & labeling
└── Klasifikasi Burnout pada Mahasiswa dan Pekerja.ipynb                   # Semua kode utama (versi Google Colab)
```

---

## ⚙️ **3. Fitur Utama Pipeline**

### ✔ **Automated labeling menggunakan SBERT**

* Menghitung similarity antar teks dengan query kategori.
* Memilih label berdasarkan skor tertinggi.
* Menerapkan threshold untuk memastikan validitas label.

### ✔ **Text Preprocessing Berlapis**

Termasuk:

* Lowercasing
* Cleaning URL
* Hapus emoji
* Hapus angka & simbol
* Remove stopwords
* Normalisasi spasi

### ✔ **Oversampling Kelas Minoritas**

Menggunakan **RandomOverSampler** agar dataset seimbang sebelum training.

### ✔ **4 Jenis Feature Extraction**

| Metode             | Deskripsi                              |
| ------------------ | -------------------------------------- |
| TF-IDF (Unigram)   | Representasi klasik berbasis frekuensi |
| TF-IDF (1–3 Gram)  | Menangkap konteks lebih panjang        |
| BoW Unigram        | Baseline sederhana                     |
| LDA Topic Features | Distribusi probabilistik topik         |

### ✔ **Deep Learning Model**

**Bidirectional LSTM** dengan:

* Embedding trainable (128 dim)
* LSTM 64 unit per arah
* Dense layer 64 neurons
* Dropout regularization

---

## 📊 **4. Hasil Evaluasi**

### **A. Classic Machine Learning**

Model terbaik berdasarkan pengujian:

| Model                 | Feature                | Akurasi |
|-----------------------|------------------------|---------|
| Naive Bayes           | Bag of Words           | 84%     |
| Naive Bayes           | TF-IDF Unigram         | 84%     |
| Naive Bayes           | TF-IDF N-grams         | 86%     |
| Naive Bayes           | LDA                    | 45%     |
| XGBoost               | Bag of Words           | 85%     |
| XGBoost               | TF-IDF Unigram         | 86%     |
| XGBoost               | TF-IDF N-grams         | 87%     |
| XGBoost               | LDA                    | 81%     |
| Logistic Regression   | Bag of Words           | 90%     |
| Logistic Regression   | TF-IDF Unigram         | 89%     |
| Logistic Regression   | TF-IDF N-grams         | 91%     |
| Logistic Regression   | LDA                    | 45%     |
| SVM                   | Bag of Words           | 91%     |
| SVM                   | TF-IDF Unigram         | 91%     |
| SVM                   | TF-IDF N-grams         | 94%     |
| SVM                   | LDA                    | 45%     |


### **B. Deep Learning**

| Model                  | Akurasi |
| ---------------------- | ------- |
| **Bidirectional LSTM** | **93%** |

---

## 🏁 **5. Kesimpulan**

Berdasarkan keseluruhan eksperimen:

* **Metode klasifikasi klasik (Random Forest + TF-IDF)** memberikan performa terbaik dengan akurasi **96%**, mengungguli pendekatan deep learning pada dataset ini.
* **Bidirectional LSTM** tetap memberikan hasil yang sangat kompetitif (**93%**) serta lebih baik dalam memahami konteks sekuensial.
* Perbedaan performa dapat dijelaskan oleh ukuran dataset, struktur teks pendek, serta keunggulan TF-IDF dalam memetakan frekuensi kata untuk masalah klasifikasi berbasis tema.

**Kesimpulan utama:**
➡ *Untuk dataset teks pendek dan jumlah data terbatas, metode klasik berbasis TF-IDF masih sangat efektif dan bahkan dapat mengungguli model deep learning yang lebih kompleks.*

---

## 🚀 **6. Cara Menjalankan**

### **1. Clone Repository**

```bash
git clone https://github.com/your-username/burnout-classification.git
cd burnout-classification
```

### **2. Jalankan Notebook**

Buka file:

```
notebook.ipynb
```

Disarankan menggunakan **Google Colab** karena membutuhkan GPU untuk BiLSTM.

---

## 🔧 **7. Teknologi yang Digunakan**

* Python 3.10+
* Sentence-BERT (`all-mpnet-base-v2`)
* Scikit-learn
* XGBoost
* TensorFlow / Keras
* NLTK
* Pandas
* Matplotlib & Seaborn

---

## 🤝 **8. Kontribusi**

Pull request sangat diterima. Anda dapat berkontribusi pada:

* Penambahan model transformer (BERT fine-tuning)
* Penambahan dashboard analisis
* Penambahan evaluasi tambahan (ROC, PR Curve, SHAP)

---
