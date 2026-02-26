# ================= ENV SAFETY =================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ================= IMPORTS =================
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import json
import re
import torch
from transformers import pipeline

torch.set_num_threads(1)

# ================= CONFIG =================
INPUT_FILE = "1form-cms1500.pdf"
COORD_FILE = "cms1500_coordinates.json"
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESS_PATH

FIELDS = [
    "patient_name",
    "insured_id",
    "insurance_plan",
    "diagnosis_code",
    "date_of_service",
    "procedure_code",
    "total_charges"
]

# ================= LOAD HF QA MODEL (SAFE) =================
print("\n[INIT] Loading HuggingFace QA model (CPU-safe)...")

qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=-1   # CPU only
)

print("[INIT] QA model loaded successfully\n")

# ================= IMAGE LOAD =================
def load_image(path):
    pages = convert_from_path(path, dpi=300)
    return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

# ================= TEMPLATE OCR =================
def template_extraction():
    img = load_image(INPUT_FILE)
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, 10
    )

    with open(COORD_FILE) as f:
        coords = json.load(f)

    def ocr(box):
        x1, y1, x2, y2 = box
        crop = thresh[
            int(y1*h):int(y2*h),
            int(x1*w):int(x2*w)
        ]
        return pytesseract.image_to_string(crop, config="--psm 6")

    raw = {k: ocr(v) for k, v in coords.items()}

    # ---------- CLEAN ----------
    def clean_name(t): return re.sub(r"[^A-Za-z ]", "", t).strip().upper()
    def clean_id(t): return re.sub(r"\D", "", t)
    def icd(t): return re.search(r"\d{3}(\.\d)?", t).group() if re.search(r"\d{3}", t) else None
    def cpt(t): return re.search(r"\d{5}", t).group() if re.search(r"\d{5}", t) else None
    def amt(t): return re.search(r"\d+(\.\d{2})?", t.replace(" ", "")).group() if re.search(r"\d", t) else None

    def date(t):
        t = t.replace("|", " ").replace("\n", " ")
        m = re.search(r"\d{2}\s+\d{2}\s+\d{2,4}", t)
        return "/".join(m.group().split()) if m else None

    def insurance(t):
        t = t.upper()
        for k in ["MEDICARE", "MEDICAID", "TRICARE", "CHAMPUS"]:
            if k in t:
                return k
        return None

    return {
        "patient_name": clean_name(raw["patient_name"]),
        "insured_id": clean_id(raw["insured_id"]),
        "insurance_plan": insurance(raw["insurance_plan"]),
        "diagnosis_code": icd(raw["diagnosis_code"]),
        "date_of_service": date(raw["date_of_service"]),
        "procedure_code": cpt(raw["procedure_code"]),
        "total_charges": amt(raw["total_charges"])
    }

# ================= FULL OCR TEXT =================
def full_ocr_text():
    img = load_image(INPUT_FILE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config="--psm 3")

# ================= HF QA FALLBACK =================
def hf_fallback(missing_fields, ocr_text):
    print("\n[HF-QA] Recovering fields:", missing_fields)

    questions = {
        "patient_name": "What is the patient's name?",
        "insured_id": "What is the insured ID number?",
        "insurance_plan": "What is the insurance plan?",
        "diagnosis_code": "What is the diagnosis code?",
        "date_of_service": "What is the date of service?",
        "procedure_code": "What is the procedure code?",
        "total_charges": "What is the total charge amount?"
    }

    result = {}
    for f in missing_fields:
        ans = qa({
            "question": questions[f],
            "context": ocr_text
        })
        result[f] = ans["answer"].strip()

    return result

# ================= MAIN =================
template = template_extraction()

print("\n[TEMPLATE RESULT]")
print(json.dumps(template, indent=4))

missing = [k for k, v in template.items() if not v]

if missing:
    ocr_text = full_ocr_text()
    recovered = hf_fallback(missing, ocr_text)
    for k in missing:
        if recovered.get(k):
            template[k] = recovered[k]

print("\n================ FINAL OUTPUT ================\n")
print(json.dumps(template, indent=4))
print("\n============================================\n")