import os
import cv2
import pickle

REAL_DIR = "SOCOFing/Real"
sift = cv2.SIFT_create()

def serialize_keypoints(keypoints):
    return [(
        kp.pt,
        kp.size,
        kp.angle,
        kp.response,
        kp.octave,
        kp.class_id
    ) for kp in keypoints]

features = []

for file_name in os.listdir(REAL_DIR):
    path = os.path.join(REAL_DIR, file_name)
    img = cv2.imread(path)
    if img is None:
        continue

    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        continue

    features.append({
        "filename": file_name,
        "keypoints": serialize_keypoints(keypoints),  # Serialize manually
        "descriptors": descriptors
    })

# Save descriptors and keypoints
with open("real_fingerprints.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Descriptors saved successfully.")
