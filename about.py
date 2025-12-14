import streamlit as st

st.title("ℹ️ About This App")
st.markdown("""
**Sistem Pengenalan Produk Ritel**  
Sistem ini dibangun menggunakan model **Vision Transformer (ViT-B/16)**
yang telah dilatih untuk klasifikasi produk ritel tertentu, serta **HSV**
untuk mendeteksi tangan berdasarkan warna kulit.

Sistem menerima masukan gambar melalui upload gambar atau capture dari
kamera, lalu mendeteksi tangan untuk memotong gambar supaya terfokus pada
produk yang dipegang.
Gambar kemudian diproses oleh model dan menampilkan produk yang dikenali.
            
**Produk yang dikenali sistem:**
""")
st.image("assets/products.png")
st.markdown("""      
**Tech Stack yang digunakan:**
- PyTorch + timm: untuk membangun dan melatih model
- OpenCV: untuk pemrosesan gambar dan deteksi warna kulit tangan
- Streamlit: untuk membangun antarmuka web  

**Developer:**
Angela Chow
""")
