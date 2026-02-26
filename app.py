import os, json, re, uuid
from flask import Flask, render_template, request, jsonify
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from transformers import pipeline

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
COORD_FILE = "cms1500_coordinates.json"
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# ================= FLASK =================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================= LOAD QA MODEL =================
print("[INIT] Loading HuggingFace QA model...")
qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=-1
)
print("[INIT] QA model loaded")

# ================= HELPERS =================
def load_image(path):
    if path.lower().endswith(".pdf"):
        pages = convert_from_path(path, dpi=300)
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)

def full_ocr_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config="--psm 3")

# ================= TEMPLATE EXTRACTION =================
def extract_template(path):
    img = load_image(path)
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
        crop = thresh[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
        return pytesseract.image_to_string(crop, config="--psm 6")

    raw = {k: ocr(v) for k, v in coords.items()}

    def clean_name(t): return re.sub(r"[^A-Za-z ]", "", t).strip().upper() or None
    def clean_id(t):
        if not t:
            return None
        return re.sub(r"[^A-Za-z0-9]", "", t).upper() or None
    def cpt(t): return re.search(r"\b\d{5}\b", t).group() if re.search(r"\b\d{5}\b", t) else None
    def amt(t): return re.search(r"\d+(\.\d{2})?", t.replace(" ", "")).group() if re.search(r"\d", t) else None
    def date(t):
        t = t.replace("|", " ")
        m = re.search(r"\d{2}\s+\d{2}\s+\d{2,4}", t)
        return "/".join(m.group().split()) if m else None
    def insurance(t):
        t = t.upper()
        if "MEDICARE" in t: return "MEDICARE"
        if "MEDICAID" in t: return "MEDICAID"
        if "TRICARE" in t or "CHAMPUS" in t: return "TRICARE"
        return None

    return {
        "patient_name": clean_name(raw["patient_name"]),
        "insured_id": clean_id(raw["insured_id"]),
        "insurance_plan": insurance(raw["insurance_plan"]),
        "diagnosis_code": None,  # recovered later
        "date_of_service": date(raw["date_of_service"]),
        "procedure_code": cpt(raw["procedure_code"]),
        "total_charges": amt(raw["total_charges"])
    }

def recover_icd(text):
    for m in re.finditer(r"\b\d{3}(?:\.\d{1,2})?\b", text):
        return m.group()
    return None

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = str(uuid.uuid4()) + "_" + file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Extract
    result = extract_template(path)
    img = load_image(path)
    ocr_text = full_ocr_text(img)

    # Recover diagnosis via regex
    result["diagnosis_code"] = recover_icd(ocr_text)

    # Save JSON
    out_file = os.path.join(
        OUTPUT_FOLDER, filename.replace(".", "_") + ".json"
    )
    with open(out_file, "w") as f:
        json.dump(result, f, indent=4)

    return jsonify({
        "status": "success",
        "data": result,
        "saved_to": out_file
    })

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)