# 🧠 Burnout Classification: Classic ML, Deep Learning, Pre-trained (Full & LoRA), and XAI

**A Comprehensive NLP Pipeline for Burnout Detection (3-Class Classification)**

Repository ini berisi implementasi *end-to-end* untuk **klasifikasi burnout** pada teks media sosial ke dalam tiga kategori:
* **College Burnout** (Burnout Mahasiswa)
* **Worker Burnout** (Burnout Pekerja)
* **No Burnout** (Tidak Burnout)

Proyek ini membandingkan berbagai pendekatan mulai dari metode tradisional hingga teknik *State-of-the-Art* yang efisien, serta dilengkapi dengan validasi transparansi model menggunakan Explainable AI (XAI).

---

## 📌 1. Pendekatan & Metodologi

Proyek ini mengeksplorasi lima paradigma pemodelan utama:

### **A. Machine Learning Klasik**
* **Fitur:** Ekstraksi fitur manual menggunakan TF-IDF (Unigram/N-gram), Bag-of-Words, dan LDA Topic Distribution.
* **Model:** Menguji algoritma Multinomial Naive Bayes, Random Forest, SVM, Logistic Regression, dan XGBoost.

### **B. Deep Learning**
* **Model:** **Bidirectional LSTM** (BiLSTM).
* **Arsitektur:** Menggunakan *trainable embedding layer* untuk menangkap konteks sekuensial dua arah (masa lalu dan masa depan) dalam kalimat.

### **C. Pre-trained Transformer (Full Fine-Tuning)**
* **Model:** **DistilBERT** (`distilbert-base-uncased`).
* **Metode:** Melatih ulang seluruh parameter model (66 juta+ parameter) untuk tugas klasifikasi spesifik. Memberikan performa terbaik namun membutuhkan sumber daya komputasi besar.

### **D. Pre-trained Transformer (LoRA - Low Rank Adaptation)**
* **Model:** DistilBERT + Adapter LoRA.
* **Metode:** Membekukan (*freeze*) model utama dan hanya melatih matriks adaptasi rank-rendah (*low-rank matrices*).
* **Keunggulan:** Meningkatkan efisiensi memori GPU secara drastis dengan performa yang tetap kompetitif (hanya melatih <1% parameter total).

### **E. Explainable AI (XAI)**
* **Metode:** **LIME** (Local Interpretable Model-agnostic Explanations).
* **Tujuan:** Memvalidasi transparansi model dengan memvisualisasikan bobot kata yang mempengaruhi keputusan prediksi (misalnya: memverifikasi apakah model menggunakan kata yang logis seperti "class" untuk mendeteksi mahasiswa).

---

## ⚙️ 2. Fitur Utama Pipeline

### ✔ Automated Labeling (SBERT)
Menggunakan **Sentence-BERT** (`all-mpnet-base-v2`) untuk memberikan label otomatis pada dataset mentah berdasarkan kemiripan semantik terhadap query target, memastikan kualitas label awal yang baik.

### ✔ Advanced Preprocessing
Meliputi *cleaning* (menghapus URL, emoji, simbol), normalisasi teks, *stopword removal*, dan penanganan ketidakseimbangan data (*imbalance*) menggunakan teknik **RandomOverSampler**.

### ✔ Feature Extraction Variatif
Pipeline mendukung berbagai metode ekstraksi fitur:
* TF-IDF Vectorizer untuk ML Klasik.
* Keras Tokenizer (Sequence Padding) untuk LSTM.
* DistilBertTokenizer (Subword Tokenization) untuk Transformer.

### ✔ Efisiensi Pelatihan (LoRA)
Implementasi konfigurasi LoRA (`r=16`, `alpha=32`, `dropout=0.1`) memungkinkan pelatihan model Transformer dilakukan pada lingkungan dengan sumber daya terbatas (seperti Google Colab T4) tanpa mengurangi akurasi secara signifikan.

### ✔ Transparansi Model (XAI)
Visualisasi interaktif menggunakan LIME untuk menyoroti kata-kata yang mendukung (hijau) atau menentang (merah) prediksi model, memastikan model bekerja dengan logika manusia yang benar.

---

## 📊 3. Hasil Evaluasi & Perbandingan

Berikut adalah ringkasan performa dari berbagai metode:

### **A. Machine Learning Klasik**
Random Forest mencatatkan performa terbaik di kategori ini.

| Model | Feature | Akurasi |
| :--- | :--- | :--- |
| Naive Bayes | TF-IDF N-grams | 86% |
| XGBoost | TF-IDF N-grams | 87% |
| Logistic Regression | TF-IDF N-grams | 91% |
| SVM | TF-IDF N-grams | 94% |
| **Random Forest** | **TF-IDF N-grams** | **96%** |

### **B. Deep Learning (BiLSTM)**
| Model | Akurasi |
| :--- | :--- |
| Bidirectional LSTM | 94% |

### **C. Pre-trained Transformer (Full Tuning vs LoRA)**
Perbandingan efisiensi dan akurasi antara metode Full Fine-Tuning dan LoRA.

| Metode | Precision | Recall | F1-Score | Akurasi | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DistilBERT (Full FT)** | 0.96 | 0.96 | 0.96 | **97%** | Akurasi Tertinggi |
| **DistilBERT + LoRA** | 0.94 | 0.94 | 0.94 | **96%** | Paling Efisien |

### **D. Analisis XAI (LIME)**
Analisis pada sampel uji (Label: *College Burnout*) menunjukkan model DistilBERT + LoRA mengambil keputusan berdasarkan fitur yang valid:

* **Probabilitas Prediksi:** 99.9% yakin *College Burnout*.
* **Top Features (Kata Kunci Berpengaruh):**
    * `class` (+0.18): Indikator kuat konteks akademik.
    * `minutes`, `hour` (+0.08): Indikator tekanan waktu/durasi.
    * `left` (+0.06): Indikator sisa energi/waktu.

---

## 🏁 4. Kesimpulan

Penelitian ini menyimpulkan tiga hasil utama:

1.  **Efisiensi Komputasi:** Metode **LoRA** terbukti berhasil melatih model bahasa besar secara efisien pada sumber daya terbatas, menghasilkan file model yang ringan namun tetap cerdas.
2.  **Performa Superior:** Kombinasi **Random Forest** dan **DistilBERT** (baik Full maupun LoRA) menunjukkan akurasi tinggi (94%-96%), membuktikan keandalannya dalam mengklasifikasikan *burnout* pada mahasiswa dan pekerja dengan tepat.
3.  **Transparansi Sistem:** Integrasi **XAI (LIME)** memvalidasi bahwa keputusan model didasarkan pada indikator linguistik yang relevan dan logis, menjadikan sistem ini dapat dipercaya (*trustworthy*) untuk implementasi nyata.

---

## 🚀 5. Cara Menjalankan (Langkah Eksekusi)

### Persiapan Library
Jalankan perintah berikut di terminal atau sel notebook untuk menginstall dependensi yang dibutuhkan:

### Library Dasar & ML Klasik
pip install scikit-learn pandas numpy nltk matplotlib seaborn imbalanced-learn

### Library Deep Learning & Transformers
pip install torch transformers accelerate

### Library Khusus LoRA (Low-Rank Adaptation)
pip install peft

### Library Khusus Explainable AI (XAI)
pip install lime

## 🚀 6. Langkah-Langkah Running Code

1.  **Clone Repository:**
    Unduh repository ini ke mesin lokal atau Google Colab Anda.

2.  **Siapkan Dataset:**
    Pastikan file dataset bernama `Burnout_3_Class_Filtered.csv` sudah berada di direktori yang sama dengan notebook.

3.  **Buka Notebook:**
    Buka file `Klasifikasi_Burnout_Lengkap.ipynb` menggunakan Jupyter Notebook, VS Code, atau Google Colab.

4.  **Eksekusi Pipeline:**
    * **Step 1:** Jalankan sel *Data Loading* dan *Preprocessing* (Cleaning, Stopword Removal).
    * **Step 2:** Jalankan sel *Oversampling* untuk menyeimbangkan kelas data.
    * **Step 3:** Pilih metode training yang diinginkan (misalnya: Bagian "Metode 6: Fine-Tuning dengan LoRA").
    * **Step 4:** Jalankan proses training (`trainer.train()`).

5.  **Evaluasi & Interpretasi:**
    * Lihat hasil *Classification Report* dan *Confusion Matrix*.
    * Jalankan bagian kode **XAI (LIME)** di akhir notebook untuk melihat visualisasi logika prediksi model pada data sampel.

---

## 🔧 7. Teknologi Utama (Tech Stack)

Proyek ini dibangun menggunakan teknologi dan pustaka berikut:

* **Bahasa Pemrograman:** Python 3.10+
* **Machine Learning Core:**
    * `scikit-learn`: Untuk TF-IDF, Naive Bayes, SVM, Random Forest, dan metrik evaluasi.
    * `xgboost`: Untuk model Gradient Boosting.
    * `imbalanced-learn`: Untuk teknik RandomOverSampler.
* **Deep Learning & NLP:**
    * `torch` (PyTorch): Backend utama untuk training model neural network.
    * `transformers` (Hugging Face): Menyediakan model DistilBERT dan Tokenizer.
    * `peft` (Hugging Face): Pustaka khusus untuk implementasi LoRA.
    * `tensorflow/keras`: Untuk implementasi model BiLSTM (opsional).
* **Explainable AI:**
    * `lime`: Untuk visualisasi interpretasi model lokal.
* **Data Processing & Visualization:**
    * `pandas` & `numpy`: Manipulasi data tabular dan array.
    * `nltk`: Pemrosesan teks dasar (stopwords).
    * `matplotlib` & `seaborn`: Visualisasi grafik dan Confusion Matrix.
