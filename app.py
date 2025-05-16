import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from shutil import copyfileobj

app = FastAPI()

REAL_DIR = "SOCOFing/Real"

@app.post("/match-fingerprint/")
async def match_fingerprint(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    with NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
        copyfileobj(file.file, tmp)
        temp_path = tmp.name

    sample = cv2.imread(temp_path)
    if sample is None:
        return JSONResponse(content={"error": "Invalid image format."}, status_code=400)

    best_score = 0
    best_filename = None
    best_image = None
    kp1, kp2, best_match_points = None, None, None

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)

    for idx, file_name in enumerate(os.listdir(REAL_DIR)):
        path = os.path.join(REAL_DIR, file_name)
        fingerprint_image = cv2.imread(path)
        if fingerprint_image is None:
            continue

        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        if descriptors_1 is None or descriptors_2 is None:
            continue

        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
        match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]

        min_keypoints = min(len(keypoints_1), len(keypoints_2))
        if min_keypoints == 0:
            continue

        score = len(match_points) / min_keypoints * 100
        if score > best_score:
            best_score = score
            best_filename = file_name
            best_image = fingerprint_image
            kp1, kp2, best_match_points = keypoints_1, keypoints_2, match_points

    os.remove(temp_path)  # Clean up

    if best_filename:
        return {
            "best_match": best_filename,
            "match_score": round(best_score, 2)
        }
    else:
        return JSONResponse(content={"message": "No match found."}, status_code=404)
