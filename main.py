from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import easyocr
import numpy as np
import cv2
import re
import torch

app = FastAPI()

# ตรวจสอบ CUDA
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# โหลด EasyOCR Reader ครั้งเดียว
reader = easyocr.Reader(['th', 'en'], gpu=torch.cuda.is_available())

@app.post("/ocr/id-card")
async def ocr_id_card(file: UploadFile = File(...)):
    image = await file.read()
    npimg = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # อ่าน OCR
    results = reader.readtext(img)

    # ดึงข้อความมา
    texts = [text[1] for text in results]

    # (ตัวอย่างง่าย) แปลงผลลัพธ์เป็น JSON
    response = {
        "full_text": " ".join(texts),
        "fields": extract_fields(texts)
    }
    return JSONResponse(content=response)

# ดึงข้อมูลเฉพาะออกมาอย่างง่าย
def extract_fields(texts):
    if isinstance(texts, list):
        text = " ".join(texts)
    else:
        text = texts

    text = text.lower().replace('\n', ' ').replace('|', '1').replace('ฺ', '').replace(']', 'l')

    data = {
        "citizen_id": None,
        "prefix": None,
        "name_th": None,
        "lastname_th": None,
        "name_en": None,
        "lastname_en": None,
        "dob": None,
        "religion": None,
        "address": None,
        "village": None,
        "subdistrict": None,
        "district": None,
        "province": None,
        "issued_date": None,
        "expired_date": None
    }

    # 1. เลขบัตรประชาชน
    cid_match = re.search(r"\b\d\s?\d{4}\s?\d{5}\s?\d{2}\s?\d\b", text)
    if cid_match:
        data["citizen_id"] = cid_match.group().replace(" ", "")

    # 2. ชื่อ-นามสกุล ภาษาไทย + คำนำหน้า
    prefixes = [
    "น\.ส\.", "นาย", "นาง",  # นามสกุลทั่วไป
    "พล\.ท\.", "พล\.ร\.ท\.", "พล\.ต\.", "พล\.ท\.",  # ตำรวจ
    "ท\.ท\.", "พล\.จ\.", "ทหาร",  # ทหาร
    "ดร\.", "ศาสตราจารย์", "หมอ", "แพทย์",  # หมอ
    
    ]

    # แก้ไข regex ให้รองรับยศทางการ
    prefix_pattern = "|".join(prefixes)
    name_th_match = re.search(rf"({prefix_pattern})\s*([ก-๙]+)\s+([ก-๙]+)", text)

    if name_th_match:
        data["prefix"] = name_th_match.group(1)
        data["name_th"] = name_th_match.group(2)
        data["lastname_th"] = name_th_match.group(3)

    # 3. ชื่อ-นามสกุล ภาษาอังกฤษ
    name_en_match = re.search(r"\b(mr|mrs|miss|dr|prof|sir|rev)[\s:]+([a-z]+)", text)
    lastname_en_match = re.search(r"last\s+name[\s:]+([a-z]+)", text)
    if name_en_match:
        data["name_en"] = name_en_match.group(2).capitalize()
    if lastname_en_match:
        data["lastname_en"] = lastname_en_match.group(1).capitalize()

    # 4. วันเกิด (ก่อนคำว่า "date of birth")
    dob_match = re.search(r"([0-9]{1,2}\s*[ก-๙a-z.]+\s*[0-9]{4}).{0,20}(เกิดวันที่|date of birth)", text)
    if dob_match:
        data["dob"] = dob_match.group(1).strip().replace('ึ', 'ิ').replace('ื', 'ิ')

    # 5. ศาสนา
    religion_match = re.search(r"ศาสนา\s*([ก-๙a-z]+)", text)
    if religion_match:
        data["religion"] = religion_match.group(1).capitalize()

    # 6. ที่อยู่ (ไม่รวมหมู่/ตำบล)
    address_match = re.search(r"(\d{1,4}\/\d{1,4}.*?)\s+(หมู่ที่|หมู่ที|หมูที่|หม่ที่|หมูที|หม่ที|หมทีหมู่|ม\.|ต\.|อ\.|จ\.)", text)
    if address_match:
        data["address"] = address_match.group(1).strip()

    # 6.1 หมู่ที่
    village_match = re.search(r"(หมู่ที่|หมู่ที|หมูที่|หม่ที่|หมูที|หม่ที|หมที\.)\s*(\d{1,2})", text)
    if village_match:
        data["village"] = village_match.group(2)

    # 6.2 ตำบล
    subdistrict_match = re.search(r"(ตำบล|ต\.)\s*([ก-๙]+)", text)
    if subdistrict_match:
        data["subdistrict"] = subdistrict_match.group(2)

    # 7. อำเภอ
    district_match = re.search(r"(อำเภอ|อ\.)\s*([ก-๙]+)", text)
    if district_match:
        data["district"] = district_match.group(2)

    # 8. จังหวัด
    province_match = re.search(r"(จังหวัด|จ\.)\s*([ก-๙]+)", text)
    if province_match:
        data["province"] = province_match.group(2)

    # 9-10. วันออกบัตร / วันหมดอายุ (issued date อยู่ก่อน "วันออกบัตร", expired อยู่หลัง)
    card_dates_match = re.search(
        r"([0-9]{1,2}\s*[ก-๙a-z.]+\s*[0-9]{4}).{0,30}วันออกบัตร.{0,30}([0-9]{1,2}\s*[ก-๙a-z.]+\s*[0-9]{4})",
        text
    )
    if card_dates_match:
        data["issued_date"] = card_dates_match.group(1).strip().replace('ึ', 'ิ').replace('ื', 'ิ')
        data["expired_date"] = card_dates_match.group(2).strip().replace('ึ', 'ิ').replace('ื', 'ิ')

    return data
