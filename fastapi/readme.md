# 🏠 House Rent Prediction API

API ini memprediksi harga sewa properti berdasarkan detail seperti jumlah kamar, ukuran, lokasi, dan preferensi penyewa. Dibangun menggunakan **FastAPI**, **scikit-learn**, dan **joblib**.

## ⚙️ Instalasi & Menjalankan API

1. **Buat virtual environment (opsional tapi disarankan)**:

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```

2. **Install dependencies**:

   ```bash
   pip install fastapi uvicorn pandas scikit-learn joblib
   ```

3. **Jalankan server FastAPI**:

   ```bash
   uvicorn main:app --reload
   ```

4. **Akses dokumentasi**:
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Redoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📬 Endpoint

### `POST /predict`

**Request Body (JSON)**:

```json
{
  "BHK": 2,
  "Size": 1200,
  "Bathroom": 2,
  "Floor": "2 out of 5",
  "AreaType": "Super Area",
  "AreaLocality": "Whitefield",
  "City": "Bangalore",
  "FurnishingStatus": "Furnished",
  "TenantPreferred": "Family",
  "PointOfContact": "Dealer"
}
```

**Response (JSON)**:

```json
{
  "predicted_rent": 24500
}
```

---

## 🛠️ Requirements

- Python 3.8+
- fastapi
- uvicorn
- pandas
- numpy
- scikit-learn
- joblib

---

## 📌 Catatan

- Pastikan file model (`.pkl`) sesuai dengan kolom yang digunakan saat pelatihan.
- Untuk performa optimal, training dan preprocessing sebaiknya dilakukan di pipeline yang konsisten.
