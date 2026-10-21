import os
import json
import csv
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
with open("routing_config.json", "r") as f:
    config = json.load(f)

CLASS_NAMES = config["class_names"]
MODEL_PATH = config["model"]["path"]
THRESHOLD = config["confidence_threshold"]
TEAM_MAP = config["team_map"]

# We must use the exact same transforms as the notebook evaluation
eval_tf = T.Compose([
    T.Grayscale(3),
    T.Resize((config["model"]["img_size"], config["model"]["img_size"])),
    T.ToTensor(),
    T.Normalize(config["model"]["imagenet_mean"], config["model"]["imagenet_std"])
])

# ==========================================
# 2. LOAD MODEL
# ==========================================
def load_model():
    print(f"Loading {config['model']['architecture']} from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Recreate the exact ResNet18 architecture from the notebook
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
    
    # Load the weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()
    return model, device

# ==========================================
# 3. ROUTING LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Route documents using ResNet18")
    parser.add_argument("--input", type=str, required=True, help="Folder containing mixed documents")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input folder '{input_dir}' does not exist.")
        return

    # Create output directories
    output_dir = Path("output")
    review_dir = output_dir / "needs_review"
    output_dir.mkdir(exist_ok=True)
    review_dir.mkdir(exist_ok=True)
    for c in CLASS_NAMES:
        (output_dir / c).mkdir(exist_ok=True)

    # Setup CSV Log
    log_file = Path("routing_log.csv")
    log_exists = log_file.exists()
    if not log_exists:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "filename", "predicted_class", "confidence", "needs_review", "team", "destination"])

    model, device = load_model()

    # Get all images in the input folder
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    
    if not files:
        print("No valid images found in input folder.")
        return

    print(f"\nProcessing {len(files)} files...\n")
    
    for file in files:
        try:
            # Preprocess
            img = Image.open(file).convert("RGB")
            tensor = eval_tf(img).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            conf, idx = probs.max(dim=0)
            
            confidence = conf.item()
            pred_class = CLASS_NAMES[idx.item()]
            needs_review = confidence < THRESHOLD
            team = "Human_Review" if needs_review else TEAM_MAP.get(pred_class, "General_Ops")
            
            # Determine destination
            if needs_review:
                dest_folder = review_dir
                flag = "REVIEW"
            else:
                dest_folder = output_dir / pred_class
                flag = "AUTO  "
                
            # Handle file collision (add timestamp or number if file exists)
            dest_path = dest_folder / file.name
            if dest_path.exists():
                dest_path = dest_folder / f"{file.stem}_{int(datetime.now().timestamp())}{file.suffix}"
            
            # Move or Copy
            if args.move:
                shutil.move(str(file), str(dest_path))
            else:
                shutil.copy2(str(file), str(dest_path))
                
            # Log to CSV
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    file.name, pred_class, round(confidence, 4),
                    needs_review, team, str(dest_path)
                ])
                
            print(f"{flag} | {confidence:.3f} | {pred_class:<22} | {file.name}")
            
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    print(f"\nDone! Check the 'output/' folder.")
    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    main()