import av
import numpy as np

from transformers import AutoImageProcessor, VideoMAEForVideoClassification
import torch
import torch.nn as nn


from utils.model_utils import read_video_pyav, sample_frame_indices



class ERVideoMAE(nn.Module):
    def __init__(self, model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics", device: str = "cuda", num_frames: int = 16, num_classes=8, **kwargs):
        """
        Initialize the ERVideoMAE model.
        Args:
            model_name (str): The name of the pretrained model.
            device (str): The device to use for computation (e.g., "cuda" or "cpu").
            num_frames (int): The number of frames to sample from the video.
        """
        super().__init__()
        self.device = device
        self.num_frames = num_frames
        self.model_name = model_name
        # VideoMAE model
        self._video_model = VideoMAEForVideoClassification.from_pretrained(model_name, **kwargs).to(device)
        self._image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Classification head
        self.linear_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self._video_model.config.num_labels, num_classes)
        ).to(device)


    def video_model(self):
        """
        Get the video model.
        Returns:
            VideoMAEModel: The video model.
        """
        return self._video_model.to("cpu")

    def image_processor(self):
        """
        Get the image processor.
        Returns:
            AutoImageProcessor: The image processor.
        """
        return self._image_processor.to("cpu")
    
    def forward(self, x):
        '''
        x: A list of images with shape: [image_0, image_2, ..., image_15]; each image is of shape (3, 224, 224)
        '''
        # after image processor, will be of shape torch.Size([1, 16, 3, 224, 224])
        # video of shape (16, 360, 640, 3) -> time, height, width, channels add to list to make [bs, 16, 360, 640, 3]
        x = self._image_processor(x, return_tensors="pt").pixel_values.to(self.device)
        x = self._video_model(x).logits
        x = self.linear_head(x)
        return x


    

