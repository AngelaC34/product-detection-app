# Panduan Menjalankan Aplikasi - Sistem Pengenalan Produk Ritel

## 1. Download Program

### Cara 1: Download ZIP dari Google Drive
Unduh file ZIP dari Google Drive.  
Link: https://drive.google.com/drive/folders/1D5qxMFYFXW3VtDhsNEne_d4iTiyIV61v?usp=drive_link

Ekstrak ZIP ke folder pilihan.

### Cara 2: Clone Repository
Pastikan Git dan Git LFS sudah terinstal.

Clone repository:  
`git clone https://github.com/AngelaC34/product-detection-app.git`

## 2. Struktur Folder

Pastikan folder berisi:  
ProductDetectionApp/  
├─ .streamlit/  
│   ├─ config.toml  
├─ assets/  
│   ├─ chiki_twist.png  
│   ├─ chitato_chijeu.png  
│   └─ ... (semua gambar)  
├─ about.py  
├─ app.py  
├─ product_detection.py  
├─ requirements.txt  
├─ tutorial.py  
├─ vit_base_patch16_224_bs16_ep50.pth  

## 3. Instalasi Paket
Pastikan Python sudah terinstal (rekomendasi Python 3.13.7).

Navigasi ke folder ProductDetectionApp.

Instal semua dependensi yang diperlukan:  
`pip install -r requirements.txt`

##  4. Menjalankan Aplikasi
Jalankan aplikasi dengan:
`streamlit run app.py`

Browser akan terbuka otomatis menampilkan aplikasi.  
Jika tidak terbuka otomatis, lihat alamat URL di terminal, biasanya:  
`http://localhost:8501`

## 5. Menggunakan Aplikasi
Pilih input metode:
- Upload untuk ambil dari perangkat.
- Capture untuk foto langsung.

(Opsional) Sesuaikan HSV di panel Adjust HSV Thresholds sesuai pencahayaan.

Klik Start Detection untuk memulai pengenalan produk.

Hasil pengenalan muncul di kolom Result, berupa gambar hasil crop dan nama produk yang dikenal.


Developer: Angela Chow (535220170)
