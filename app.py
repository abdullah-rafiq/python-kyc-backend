import os
os.environ["OPENCV_DISABLE_DNN_TYPING"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import io
import json
import base64
import sys
import re
import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
from deepface import DeepFace
from PIL import Image

app = Flask(__name__)

# -------------------- Model Initialization --------------------
print("Loading KYC Models...", file=sys.stderr)

# OCR
reader = easyocr.Reader(['en', 'ur'], gpu=False)

# YOLO
yolo_model = YOLO(os.getenv("YOLO_MODEL", "yolov8n.pt"))


print("Models loaded!", file=sys.stderr)

URDU_RE = re.compile(r'[\u0600-\u06FF]')

# -------------------- Helper Functions --------------------

def decode_image(image_input):
    try:
        if isinstance(image_input, str) and "," in image_input:
            image_input = image_input.split(",")[1]

        img_bytes = base64.b64decode(image_input) if isinstance(image_input, str) else image_input
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return img
    except Exception as e:
        raise ValueError(f"Invalid image input: {e}")

def crop_card(image):
    results = yolo_model(image, verbose=False)
    best_box = None
    max_area = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (int(x1), int(y1), int(x2), int(y2))
    if best_box:
        x1, y1, x2, y2 = best_box
        h, w, _ = image.shape
        x1 = max(0, x1 - 10); y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10); y2 = min(h, y2 + 10)
        return image[y1:y2, x1:x2]
    return image

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def extract_cnic_info(text_lines):
    info = {
        "fullName": None,
        "fatherName": None,
        "cnicNumber": None,
        "dateOfBirth": None,
        "dateOfIssue": None,
        "dateOfExpiry": None,
        "addressUrdu": {"line1": None, "line2": None, "district": None, "tehsil": None},
    }
    if not text_lines:
        return info
    lines = [str(t or "").strip() for t in text_lines if str(t or "").strip()]
    joined = " ".join(lines)
    m = re.search(r"\b\d{5}-\d{7}-\d\b", joined)
    if m: info["cnicNumber"] = m.group(0)
    date_pattern = r"\b(\d{2}[-/\.]\d{2}[-/\.]\d{4})\b"
    dates = re.findall(date_pattern, joined)
    if dates:
        if len(dates) >= 1: info["dateOfBirth"] = dates[0]
        if len(dates) >= 2: info["dateOfIssue"] = dates[1]
        if len(dates) >= 3: info["dateOfExpiry"] = dates[2]

    HEADER_PHRASES = ["PAKISTAN","ISLAMIC REPUBLIC OF PAKISTAN","ISLAMIIC REPUBLIC OF PAKISTAN","NATIONAL IDENTITY CARD","GOVERNMENT OF PAKISTAN"]
    def is_header_line(s): return any(phrase in s.upper() for phrase in HEADER_PHRASES)

    raw_candidates = []
    for ln in lines:
        if not ln or is_header_line(ln) or not any(ch.isalpha() for ch in ln) or sum(ch.isdigit() for ch in ln) > 2: continue
        if len(ln.split()) >= 2: raw_candidates.append(ln)

    latin_candidates = [ln for ln in raw_candidates if not URDU_RE.search(ln)]
    urdu_candidates = [ln for ln in raw_candidates if URDU_RE.search(ln)]

    info["fullName"] = latin_candidates[0] if latin_candidates else (raw_candidates[0] if raw_candidates else None)
    father = None
    if len(latin_candidates) >= 2: father = latin_candidates[1]
    else:
        for ln in raw_candidates:
            if ln != info["fullName"]: father = ln; break
    info["fatherName"] = father

    urdu_lines = [ln for ln in lines if URDU_RE.search(ln)]
    if urdu_lines:
        info["addressUrdu"]["line1"] = urdu_lines[0]
        if len(urdu_lines) > 1: info["addressUrdu"]["line2"] = urdu_lines[1]

    return info

# -------------------- Endpoints --------------------

@app.route('/', methods=['GET'])
def health(): return "KYC Engine is Running"

@app.route('/verify-cnic', methods=['POST'])
def verify_cnic():
    try:
        data = request.json or {}
        image_data = data.get('image')
        if not image_data: return jsonify({"error": "No image provided", "errorCode": "NO_IMAGE"}), 400
        img = decode_image(image_data)
        cropped = crop_card(img)
        ocr_image = preprocess_for_ocr(cropped)
        text_results = reader.readtext(ocr_image, detail=0)
        extracted = extract_cnic_info(text_results)
        return jsonify({"rawText": text_results, **extracted})
    except Exception as e:
        return jsonify({"error": str(e), "errorCode": "CNIC_INTERNAL_ERROR"}), 500

@app.route('/face-verify', methods=['POST'])
def face_verify():
    try:
        data = request.json or {}
        img1_data = data.get('image1')
        img2_data = data.get('image2')
        if not img1_data or not img2_data: 
            return jsonify({"error": "image1 and image2 required","errorCode":"MISSING_IMAGES"}), 400
        img1 = decode_image(img1_data)
        img2 = decode_image(img2_data)
        result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name='VGG-Face', enforce_detection=False)
        distance = float(result.get('distance',0.0))
        threshold = float(result.get('threshold',1.0) or 1.0)
        verified = bool(result.get('verified', False))
        confidence = max(0.0, min(1.0, 1.0 - (distance/threshold))) if threshold>0 else 0.0
        return jsonify({"verified":verified,"distance":distance,"threshold":threshold,"confidence":confidence})
    except Exception as e:
        return jsonify({"error": str(e),"errorCode":"FACE_INTERNAL_ERROR"}),500

@app.route('/shop-verify', methods=['POST'])
def shop_verify():
    try:
        data = request.json or {}
        image_data = data.get('image')
        if not image_data: return jsonify({"error": "No image provided", "errorCode": "NO_IMAGE"}), 400
        img = decode_image(image_data)
        results = yolo_model(img, verbose=False)
        detected_objects=[]
        for r in results:
            for box in r.boxes:
                cls_id=int(box.cls[0]); cls_name=yolo_model.names[cls_id]; conf=float(box.conf[0])
                if conf>0.4: detected_objects.append(cls_name)
        ocr_text=reader.readtext(img, detail=0)
        unique_objects=sorted(set(detected_objects))
        has_person='person' in unique_objects
        object_factor=min(1.0,len(unique_objects)/5.0)
        text_factor=0.3 if isinstance(ocr_text,list) and len(ocr_text)>3 else 0.0
        score=max(0.0, min(1.0, object_factor+text_factor+(0.2 if has_person else 0.0)))
        notes=[]
        if not unique_objects: notes.append("no_objects_detected")
        if not has_person: notes.append("no_person_detected")
        if not ocr_text: notes.append("no_text_detected")
        return jsonify({"detected_objects":unique_objects,"text_content":ocr_text,"score":score,"notes":notes})
    except Exception as e:
        return jsonify({"error": str(e),"errorCode":"SHOP_INTERNAL_ERROR"}),500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
