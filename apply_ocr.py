# apply_ocr.py
import cv2
import json
import argparse
import os
import pytesseract

parser = argparse.ArgumentParser(description="Aplică OCR folosind ROI din master.")
parser.add_argument("--master", required=True, help="Calea către imaginea master.")
parser.add_argument("--prepared", required=True, help="Calea către imaginea pregătită.")
args = parser.parse_args()

master_img_path = args.master
prepared_img_path = args.prepared

# Derivăm calea fișierului JSON corespunzător ROI-urilor
json_path = master_img_path.replace(".png", ".json")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"❌ Nu există fișierul ROI: {json_path}")

# Citim imaginile
master_img = cv2.imread(master_img_path)
prepared_img = cv2.imread(prepared_img_path)

# Calculăm factorii de scalare între master și imaginea pregătită
h_master, w_master = master_img.shape[:2]
h_prepared, w_prepared = prepared_img.shape[:2]
scale_x = w_prepared / w_master
scale_y = h_prepared / h_master

# Citim ROI-urile
with open(json_path, "r", encoding="utf-8") as f:
    roi_dict = json.load(f)

# Folder pentru debug
os.makedirs("debug", exist_ok=True)

results = {}

for roi in roi_dict:
    key = roi.get("nume", "unknown")
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    # Aplicăm scalarea în funcție de imaginea pregătită
    x_scaled = int(x * scale_x)
    y_scaled = int(y * scale_y)
    w_scaled = int(w * scale_x)
    h_scaled = int(h * scale_y)

    # Crop ROI
    crop = prepared_img[y_scaled:y_scaled + h_scaled, x_scaled:x_scaled + w_scaled]
    if crop.size == 0:
        print(f"⚠️ ROI gol pentru cheia {key} — poate e în afara imaginii.")
        results[key] = ""
        continue

    # Conversie în grayscale + medie blur pentru reducere noise
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    filtered = cv2.medianBlur(gray, 3)

    # Salvăm imaginea pentru debug
    debug_path = os.path.join("debug", f"{key}.png")
    cv2.imwrite(debug_path, filtered)

    # OCR
    config = "--psm 6"  # PSM 7: line of text
    text = pytesseract.image_to_string(filtered, config=config, lang="por").strip()
    results[key] = text

# Exportăm rezultatele în JSON
output_json = "ocr_result.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ OCR finalizat. Rezultatele au fost salvate în {output_json}")
