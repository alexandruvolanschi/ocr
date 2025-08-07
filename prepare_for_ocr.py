# prepare_for_ocr.py
import cv2
import numpy as np
import argparse
import subprocess
import os

def autocrop(image, threshold=10):
    """Taie margini goale (aproape negre sau albe) din jurul imaginii."""
    mask = image > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image  # fallback – nu tăiem nimic
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]

def sharpen(image):
    """Aplică un sharpening ușor (kernel clasic)."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

parser = argparse.ArgumentParser(description="Pregătește imaginea pentru OCR.")
parser.add_argument("--master", required=True, help="Imaginea master (ex: master_1.png)")
parser.add_argument("--input", required=True, help="Imaginea corectată (ex: output_homography_test.png)")
args = parser.parse_args()

INPUT_IMAGE = args.input
MASTER_IMAGE = args.master
OUTPUT_IMAGE = "output_auto_ocr_ready.png"

print("📥 Citim imaginea de input...")
img = cv2.imread(INPUT_IMAGE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("✂️ Aplicăm autocrop pe marginile goale...")
gray = autocrop(gray)

# Histograma imaginii
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
total_pixels = gray.shape[0] * gray.shape[1]
dark_pixels = np.sum(hist[:50])
bright_pixels = np.sum(hist[205:])

dark_ratio = dark_pixels / total_pixels
bright_ratio = bright_pixels / total_pixels

if dark_ratio > 0.4:
    print("🕶️ Imagine prea întunecată — clahe agresiv")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
elif bright_ratio > 0.4:
    print("💡 Imagine prea luminoasă — reducere + clahe")
    corrected = cv2.convertScaleAbs(gray, alpha=0.9, beta=-30)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(corrected)
else:
    print("⚖️ Imagine echilibrată — clahe standard")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

# ... (restul codului e identic până la partea cu noise + sharpen)

print("🧽 Noise reduction ușor... (median blur)")
gray = cv2.medianBlur(gray, 3)

# Evaluăm claritatea
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
print(f"🔬 Claritate estimată (Laplacian): {laplacian_var:.2f}")

print("🔎 Sharpening pentru claritate...")
gray = sharpen(gray)

if laplacian_var < 100:
    print("🧠 Claritate slabă detectată — aplicăm sharpen suplimentar.")
    gray = sharpen(gray)

# === TEST: Dacă vrei threshold pentru texte foarte grase, activează linia de mai jos
# gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                              cv2.THRESH_BINARY, blockSize=15, C=10)

cv2.imwrite(OUTPUT_IMAGE, gray)
print(f"✅ Imagine pregătită pentru OCR: {OUTPUT_IMAGE}")

print("🚀 Trecem la pasul 3 – Aplicare OCR...")
subprocess.run([
    "python",
    "apply_ocr.py",
    "--master", MASTER_IMAGE,
    "--prepared", OUTPUT_IMAGE,
#    "--prepared", "output_homography_test.png"
])
