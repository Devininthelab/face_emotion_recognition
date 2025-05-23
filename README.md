# Emotion recognition
## Overview

Using VideoMAE preatrained for emotion recognition classification.

## Features

- Uniformly divide an audio clip into 16 parts.
- EXtract MFCC + facial frame for each part

## Installation

```bash
git clone git@github.com:Devininthelab/face_emotion_recognition.git
cd emotion_recognition
pip install -r requirements.txt
```

## Usage

1. Fisrt run extract feature under main.py to extracted all the features 
2. For training, run this command

```bash
python main.py --epochs 50 --batch_size 16 --num_workers 2 --seed 42 --learning_rate 1e-4 --weight_decay 0.05 --save_path ./checkpoints --run_name random_shuffle
```

