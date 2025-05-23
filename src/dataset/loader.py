import os
import json
from typing import List, Tuple

import numpy as np
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch
import random


@dataclass
class FeatureSample:
    feature_path: str
    label: int

def collate_fn(batch):
        """
        Return a batch for now
        """
        return batch

class EmotionDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples: List[FeatureSample] = []

        # Load all lines from jsonl
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(FeatureSample(**data))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        feature_path = sample.feature_path
        label = sample.label

        # Load all images in the folder. Perform FSLF first
        image_files = sorted([
            f for f in os.listdir(feature_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ], reverse=False) # SETTING reverse to True to FSLF; 

        ####################
        # Perform OFOS: one facial one spectrogram
        # all_frames = sorted([f for f in image_files if f.startswith("frame_")], key=lambda x: int(x.split("_")[1].split(".")[0]))
        # all_mfccs = sorted([f for f in image_files if f.startswith("mfcc_")], key=lambda x: int(x.split("_")[1].split(".")[0]))

        # interleaved_files = []
        # for f, m in zip(all_frames, all_mfccs):
        #     interleaved_files.extend([f, m])

        # image_files = interleaved_files
        #####################

        # Random shufle the image files
        random.shuffle(image_files)

        images: List[np.ndarray] = []
        for img_file in image_files:
            img_path = os.path.join(feature_path, img_file)
            with Image.open(img_path) as img:
                img_np = np.array(img.convert("RGB"))
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # C, H, W
                images.append(img_tensor)

        return images, label
    
def collate_fn(batch):
    images_batch, labels_batch = zip(*batch)
    return list(images_batch), torch.tensor(labels_batch)
    
    


