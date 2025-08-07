import cv2
import numpy as np
import pytesseract
import os
import subprocess

from skimage.metrics import structural_similarity as ssim

# === CONFIG ===
MASTER_FOLDER = r"mastere_cropped"
INPUT_PATH = 'input.png'
OUTPUT_PATH = 'output_homography_test.png'
SCALE_UP = 1.8  # +80% creștere rezoluție

MASTER_PATHS = [
    os.path.join(MASTER_FOLDER, f)
    for f in os.listdir(MASTER_FOLDER)
    if f.lower().endswith(".png")
]

def load_gray(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

def resize_to_match(template, target):
    return cv2.resize(template, (target.shape[1], target.shape[0]))

def compute_template_score(template, input_img):
    template_resized = resize_to_match(template, input_img)
    result = cv2.matchTemplate(input_img, template_resized, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val

def compute_color_score(template_path, input_path):
    template = cv2.imread(template_path)
    input_img = cv2.imread(input_path)

    template_resized = cv2.resize(template, (input_img.shape[1], input_img.shape[0]))
    diff = cv2.absdiff(template_resized, input_img)
    score = 1.0 - (np.mean(diff) / 255.0)
    return max(0.0, min(1.0, score))

def extract_words(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    words = set([w.lower() for w in text.split() if len(w) > 1])
    return words

def compute_ocr_score(master_words, input_words):
    if not master_words or not input_words:
        return 0.0
    common = master_words & input_words
    union = master_words | input_words
    return len(common) / len(union)

# === PRELOAD ===
input_gray = load_gray(INPUT_PATH)
input_words = extract_words(INPUT_PATH)

template_scores = []
color_scores = []
ocr_scores = []

for master_path in MASTER_PATHS:
    master_gray = load_gray(master_path)
    template_score = compute_template_score(master_gray, input_gray)
    color_score = compute_color_score(master_path, INPUT_PATH)

    master_words = extract_words(master_path)
    ocr_score = compute_ocr_score(master_words, input_words)

    template_scores.append(template_score)
    color_scores.append(color_score)
    ocr_scores.append(ocr_score)

# === CALCUL PONDERI ===
def dynamic_weights(*score_lists):
    deltas = [max(s) - min(s) for s in score_lists]
    total = sum(deltas)
    if total == 0:
        return [1 / len(deltas)] * len(deltas)
    return [d / total for d in deltas]

w_template, w_color, w_ocr = dynamic_weights(template_scores, color_scores, ocr_scores)

# === SCOR FINAL ===
final_scores = []
for i in range(len(MASTER_PATHS)):
    final = (
        w_template * template_scores[i] +
        w_color * color_scores[i] +
        w_ocr * ocr_scores[i]
    )
    final_scores.append(final)

best_idx = int(np.argmax(final_scores))
best_match_path = MASTER_PATHS[best_idx]

print(f"✅ Masterul ales: {os.path.basename(best_match_path)}")

# === HOMOGRAFIE ===
master_img = cv2.imread(best_match_path)
input_img = cv2.imread(INPUT_PATH)

# downscale master pentru ORB doar pt keypoints
scale = 1000 / max(master_img.shape[:2])
master_scaled = cv2.resize(master_img, None, fx=scale, fy=scale)

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(master_scaled, None)
kp2, des2 = orb.detectAndCompute(input_img, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    # compensăm pentru scalare
    src_pts *= 1 / scale

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if M is not None:
        h, w = master_img.shape[:2]
        warped = cv2.warpPerspective(input_img, M, (w, h))

        # === ⬆️ SCALE UP (rezoluție îmbunătățită) ===
        new_w = int(warped.shape[1] * SCALE_UP)
        new_h = int(warped.shape[0] * SCALE_UP)
        enhanced = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(OUTPUT_PATH, enhanced)
        print(f"✅ Imaginea finală (îmbunătățită) a fost salvată ca '{OUTPUT_PATH}'")

        # === Lansăm PASUL 2 automat ===
        subprocess.run([
            "python", "prepare_for_ocr.py",
            "--master", best_match_path,
            "--input", OUTPUT_PATH
        ])
    else:
        print("❌ Nu s-a putut calcula matricea de transformare.")
else:
    print("❌ Prea puține potriviri pentru homografie.")
