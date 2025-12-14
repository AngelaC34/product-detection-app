import streamlit as st

st.title("How to Use This App❓")
st.markdown("""
**Langkah-langkah penggunaan aplikasi:**  
1. Pilih cara memasukkan gambar dengan memilih:
    - **Upload**: Klik **"Browse files"** untuk mengunggah gambar dari perangkat.
    - **Capture**: Allow access kepada kamera dan klik **"Take Photo"** untuk mengambil gambar langsung menggunakan kamera perangkat atau webcam.
2. (Opsional) Buka panel **"Adjust HSV Thresholds"** untuk menyesuaikan HSV dengan pencahayaan tempat gambar diambil.
    Nilai HSV dapat disesuaikan menggunakan slider yang tersedia.
3. Klik tombol **"Start Detection"** untuk memulai proses deteksi produk.
4. Hasil deteksi akan ditampilkan di kolom **"Result"** sebelah kanan, termasuk gambar yang telah dipotong dan daftar produk yang dikenali.
            
**Catatan:**
- Pastikan produk pada gambar termasuk dalam produk yang dikenali oleh sistem.
- Pastikan gambar diambil dari jarak yang tidak terlalu dekat atau terlalu jauh, sekitar 1-2 meter dari produk.
- Pastikan pencahayaan tempat gambar diambil cukup baik.
- Pastikan area kulit terbesar pada gambar merupakan area tangan, bukan wajah atau bagian tubuh lainnya.
- Pastikan latar belakang gambar tidak memiliki warna yang mirip dengan warna kulit.
""")
