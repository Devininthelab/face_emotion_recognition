from models.videomae import ERVideoMAE
from models.trainer import Trainer
import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import json
import argparse
import logging

from preprocess.audio_feature import extract_frames_and_mfccs
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from dataset.loader import EmotionDataset, collate_fn
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ERVideoMAE(model_name="MCG-NJU/videomae-base-finetuned-kinetics", device=device, num_frames=16, num_classes=8).to(device)
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




# for i, (videos, labels) in enumerate(tqdm(trainloader)):
#     # images: List of batches of images
#     # labels: List of corresponding labels
#     # Process the images and labels as needed
    
#     labels.to(device)
#     with torch.no_grad():
#         logits = models(videos)
#         print(logits.shape)  # (batch_size, num_classes)
#         predicted_label = logits.argmax(dim=-1)
#         print(predicted_label)
#         print(logits)
#     break

# print(model.__class__.__name__)
# print(dataset.__class__.__name__)

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Emotion recognition for RAVDNESS.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-2)  
    parser.add_argument("--weight_decay", type=float, default=0.05)  
    parser.add_argument("--save_path", type=str, default="./checkpoints")   
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_name", type=str, default="run")

    args = parser.parse_args()
    set_seed(args.seed)
    log_file = f"{args.run_name}.log"
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
    )
    logging.info("Loading dataset...")
    
    
    dataset = EmotionDataset(config["extracted_feature_metadata"])

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    logging.info(f"Train size: {len(train_dataset)}")
    logging.info(f"Test size: {len(test_dataset)}")
    logging.info(f"Starting training ..... ")

    model = ERVideoMAE(model_name="MCG-NJU/videomae-base-finetuned-kinetics", device=device, num_frames=16, num_classes=8).to(device)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        num_epochs=args.epochs,
        device=device,
        save_path=args.save_path,
        run_name=args.run_name
    )

    trainer.train()

    logging.info("Training complete.")



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #create_features()
    main() # python main.py --epochs 50 --batch_size 16 --num_workers 2 --seed 42 --learning_rate 1e-4 --weight_decay 0.05 --save_path ./checkpoints --run_name ffls

    
    