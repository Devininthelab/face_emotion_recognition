from models.videomae import ERVideoMAE
import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import json

from preprocess.audio_feature import extract_frames_and_mfccs
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from dataset.loader import EmotionDataset, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = ERVideoMAE(model_name="MCG-NJU/videomae-base-finetuned-kinetics", device=device, num_frames=16, num_classes=8).to(device)
# videos = []
# for i in range(16):
#     video = list(torch.randint(0, 256,  (16, 3, 13, 57), dtype=torch.uint8))
#     videos.append(video)



# with torch.no_grad():
#     logits = models(videos)
#     print(logits.shape)  # (batch_size, num_classes)
#     predicted_label = logits.argmax(dim=-1)
#     print(predicted_label)
#     print(logits)

##################################################################    
# Loading config files:
with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
##################################################################

def create_features():
    if os.path.exists(config["feature_dir"]):
        print(f"Loading features from {config['feature_dir']}")
        return
    else:
        print(f"Extracting features to {config['feature_dir']}") 
        os.makedirs(config["feature_dir"], exist_ok=True)


    meta_data = config["meta_data"]

    video_paths = []
    labels = []
    with open(meta_data, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            if obj["modality"] == 1: # videos contains both visual and audio
                video_paths.append(obj["path"])
                labels.append(obj["label"])


    os.makedirs(os.path.dirname(config["extracted_feature_metadata"]), exist_ok=True) if os.path.dirname(config["extracted_feature_metadata"]) else None

    with open(config["extracted_feature_metadata"], 'w', encoding='utf-8') as meta_file:
        for p, l in tqdm(zip(video_paths, labels), total=len(video_paths)):
            video_path = p
            label = l

            file_name = os.path.split(video_path)[-1].split(".")[0]
            feature_path = os.path.join(config["feature_dir"], file_name)
            if os.path.exists(feature_path):
                print(f"Feature already exists for {file_name}, skipping...")
                continue
            os.makedirs(feature_path, exist_ok=True)    
            extract_frames_and_mfccs(video_path, feature_path, num_segments=8)

            meta_entry = {
                    "feature_path": feature_path,
                    "label": label
                }
            meta_file.write(json.dumps(meta_entry) + '\n')

dataset = EmotionDataset(config["extracted_feature_metadata"])
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

for i, (videos, labels) in enumerate(tqdm(dataloader)):
    # images: List of batches of images
    # labels: List of corresponding labels
    # Process the images and labels as needed
    
    labels.to(device)
    with torch.no_grad():
        logits = models(videos)
        print(logits.shape)  # (batch_size, num_classes)
        predicted_label = logits.argmax(dim=-1)
        print(predicted_label)
        print(logits)
    break