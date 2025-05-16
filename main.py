import os
import cv2
import numpy as np
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from shutil import copyfileobj

app = FastAPI()

# Load precomputed descriptors and keypoints
with open("real_fingerprints.pkl", "rb") as f:
    stored_features = pickle.load(f)

sift = cv2.SIFT_create()

@app.post("/match-fingerprint/")
async def match_fingerprint(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
        copyfileobj(file.file, tmp)
        temp_path = tmp.name

    sample = cv2.imread(temp_path)
    os.remove(temp_path)  # Clean up

    if sample is None:
        return JSONResponse(content={"error": "Invalid image format."}, status_code=400)

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    if descriptors_1 is None:
        return JSONResponse(content={"error": "No descriptors found in uploaded image."}, status_code=400)

    best_score = 0
    best_match = None

    matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})

    for feature in stored_features:
        descriptors_2 = feature["descriptors"]
        try:
            matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
        except cv2.error:
            continue

        match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]

        # Use min keypoints for percentage calculation
        min_kp = min(len(descriptors_1), len(descriptors_2))
        if min_kp == 0:
            continue

        score = len(match_points) / min_kp * 100

        if score > best_score:
            best_score = score
            best_match = feature["filename"]

    if best_match:
        return {
            "best_match": best_match,
            "match_score": round(best_score, 2)
        }
    else:
        return JSONResponse(content={"message": "No match found."}, status_code=404)
