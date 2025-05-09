# Thai National ID Card OCR API (FastAPI + EasyOCR)

This project is a REST API built using **FastAPI**, designed to extract information from **Thai National ID cards** using **EasyOCR**. It supports both **Thai and English** text, and can utilize **GPU acceleration** via PyTorch (if available).

---

## 🚀 Features

- 🧠 OCR extraction using EasyOCR (`th` + `en`)
- ⚡️ GPU support via PyTorch and CUDA (if available)
- 📄 Extracts structured fields like:
  - Citizen ID
  - Thai & English names
  - Date of birth
  - Religion
  - Address (Village, Subdistrict, District, Province)
  - Card issued/expired dates

---

## 📦 Requirements

- Python 3.8+
- pip
- (Optional) NVIDIA GPU with CUDA support

---

## 🛠 Installation

```bash
# 1. Clone this repo
git clone https://github.com/RikuAlice01/THAI-OCR.git
cd THAI-OCR

# 2. Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt
````

---

## 🧪 Run the API

### 🖥 CPU-only:

```bash
uvicorn main:app --reload
```

### 🚀 With GPU support (multi-worker):

```bash
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

---

## 📤 API Usage

### `POST /ocr/id-card`

**Request:**

* `multipart/form-data` with image file (`.jpg`, `.png`, etc.)

**Example using `curl`:**

```bash
curl -X POST "http://localhost:8000/ocr/id-card" -F "file=@/path/to/idcard.jpg"
```

**Response (JSON):**

```json
{
  "full_text": "Mr Somchai...1234567890123...",
  "fields": {
    "citizen_id": "1234567890123",
    "prefix": "นาย",
    "name_th": "สมชาย",
    "lastname_th": "ใจดี",
    "name_en": "Somchai",
    "lastname_en": "Jaidee",
    "dob": "1 มกราคม 2525",
    "religion": "พุทธ",
    "address": "123/45",
    "village": "2",
    "subdistrict": "ตำบล",
    "district": "เมือง",
    "province": "เชียงใหม่",
    "issued_date": "1 มกราคม 2563",
    "expired_date": "1 มกราคม 2573"
  }
}
```

---

## 🧠 Notes

* You can adjust regex or text-cleaning rules in `extract_fields()` to improve accuracy.
* OCR may vary depending on image quality and resolution.

---

## ⚙️ Tech Stack

* [FastAPI](https://fastapi.tiangolo.com/)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [PyTorch](https://pytorch.org/)
* [Uvicorn](https://www.uvicorn.org/)

---

## 📄 License

MIT License – feel free to use, modify, and contribute.

---

## 🙋‍♂️ Author

Sitthichai S.
