import os
import cv2
import torch
import numpy as np
import argparse
import glob
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import csv

# ---------------- CONFIG ---------------- #
MODEL_NAME = "facebook/dino-vits16"
FRAME_SIZE = (224, 224)  # Resize to match ViT input
SIM_THRESHOLD = 0.98     # Cosine similarity threshold
KEYFRAME_DIR = "keyframes"
MAP_DIR = "map"
# ---------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINO model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def extract_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

def extract_unique_frames(video_path, keyframe_root, map_root):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    prev_embedding = None

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(keyframe_root, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)  # ⏱️ Get video FPS

    csv_path = os.path.join(map_root, f"{video_name}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "pts_time", "fps", "frame_idx"])  # 📝 Updated header

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img).resize(FRAME_SIZE)

            emb = extract_embedding(img_pil)

            is_unique = True
            if prev_embedding is not None:
                sim = cosine_similarity([emb], [prev_embedding])[0][0]
                if sim >= SIM_THRESHOLD:
                    is_unique = False

            if is_unique:
                prev_embedding = emb
                frame_filename = f"{count}.jpg"
                save_path = os.path.join(video_out_dir, frame_filename)
                cv2.imwrite(save_path, frame)

                # ⏱️ Get timestamp in seconds
                pts_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # 📝 Write to CSV (all columns as int)
                writer.writerow([
                    int(saved),
                    int(pts_time),
                    int(fps),
                    int(count)
                ])
                saved += 1

            count += 1

    cap.release()
    return saved

def process_videos(input_path):
    video_paths = []
    if os.path.isdir(input_path):
        video_paths = glob.glob(os.path.join(input_path, "*.mp4"))
    elif os.path.isfile(input_path):
        video_paths = [input_path]
    else:
        print("❌ Invalid input path.")
        return

    os.makedirs(KEYFRAME_DIR, exist_ok=True)
    os.makedirs(MAP_DIR, exist_ok=True)

    for path in tqdm(video_paths, desc="🚀 Processing videos"):
        print(f"\n📹 Processing: {os.path.basename(path)}")
        saved = extract_unique_frames(path, KEYFRAME_DIR, MAP_DIR)
        print(f"✅ Saved {saved} unique frames from {os.path.basename(path)}")

# ------------------ CLI ------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to video file or folder containing videos")
    args = parser.parse_args()

    process_videos(args.input)
