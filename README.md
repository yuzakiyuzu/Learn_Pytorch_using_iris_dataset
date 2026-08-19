# Learn PyTorch – Simple Neural Network (Iris Dataset)

Halo 👋

Repository ini berisi **contoh proyek Machine Learning sederhana menggunakan PyTorch** untuk klasifikasi dataset **Iris**. Proyek ini cocok untuk pemula yang ingin memahami:

* Dasar Neural Network di PyTorch
* Alur training model (forward, loss, backward, optimizer)
* Evaluasi model
* Visualisasi loss menggunakan Matplotlib



---

## 📂 Struktur File

```
.
├── Learn-Pytorch-Simple_NN.py   # Script utama training & evaluasi model
├── IRIS.csv                    # Dataset Iris
├── requirements.txt            # Dependency Python
└── README.md                   # Dokumentasi proyek
```

---

## 🧠 Apa yang Dibuat di Proyek Ini?

### 1. Model Neural Network

Model dibuat menggunakan `torch.nn.Module` dengan arsitektur:

* Input layer: 4 fitur (sepal & petal)
* Hidden layer 1: 8 neuron
* Hidden layer 2: 9 neuron
* Output layer: 3 kelas (setosa, versicolor, virginica)

Aktivasi yang digunakan: **ReLU**

---

### 2. Dataset & Preprocessing

* Dataset dibaca menggunakan **pandas**
* Label (`species`) diubah dari string ke integer menggunakan `LabelEncoder`
* Data dikonversi ke **Tensor PyTorch**
* Data dibagi menjadi:

  * 80% training
  * 20% testing

---

### 3. Training Model

* Loss Function: `CrossEntropyLoss`
* Optimizer: `Adam`
* Learning Rate: `0.01`
* Epoch: `100`

Selama training:

* Model melakukan forward pass
* Loss dihitung
* Gradien di-reset
* Backpropagation dijalankan
* Bobot diperbarui
* Nilai loss disimpan ke list

---

### 4. Visualisasi Loss

Setelah training, loss diplot menggunakan **Matplotlib** untuk melihat:

* Apakah model belajar
* Apakah loss menurun seiring epoch

Grafik ini menunjukkan proses belajar model dari error besar → kecil 📉

---

### 5. Evaluasi Model

Evaluasi dilakukan menggunakan:

```python
with torch.no_grad():
```

Artinya:

* Model **tidak belajar**
* Lebih hemat memori
* Cocok untuk inference / testing

Model diuji pada data test dan:

* Loss evaluasi dihitung
* Prediksi dibandingkan dengan label asli
* Jumlah prediksi benar dihitung

Output terminal akan menampilkan:

```
[index] tensor(prediksi) | label_asli | hasil_prediksi
```

---

## 📊 Contoh Output

* Grafik loss vs epoch
* Log training setiap 10 epoch
* Hasil prediksi per data test
* Total prediksi benar

---

## ⚙️ Cara Menjalankan Proyek

### 1. Install Dependency

```bash
pip install -r requirements.txt
```

### 2. Jalankan Script

```bash
python Learn-Pytorch-Simple_NN.py
```

Jika GPU tersedia, model otomatis menggunakan **CUDA** 🚀

---

## 🎯 Tujuan Proyek

Proyek ini dibuat untuk:

* Belajar PyTorch dari nol
* Memahami konsep `epoch`, `loss`, `optimizer`
* Latihan klasifikasi
* Dasar sebelum lanjut ke CNN / proyek AI lebih kompleks

---

## 🧩 Next Step (Ide Lanjutan)

* Tambah validasi accuracy (%)
* Simpan & load model (`torch.save`)
* Coba dataset lain
* Visualisasi confusion matrix
* Upgrade ke GUI / aplikasi kecil

---

## 🐣 Catatan

Proyek ini **fokus ke pemahaman konsep**, bukan optimasi maksimal.
Koding dibuat sengaja sederhana dan mudah dibaca.

Kalau kamu baru belajar PyTorch → ini starting point yang solid 💪

---


