import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import torch.nn as nn
import torch

from bbox_viewer import show_tracked_video
from database import VideosDatabase
import numpy as np
import cv2
from PIL import Image, ImageOps

#VIDEOS PREPROCESSING:
# 1) YOLOv8 automatically performs several preprocessing steps, including conversion to RGB,
# scaling pixel values to the range [0, 1], and normalization using predefined mean and standard deviation values.
# 2)

""""class GaussianBlur:
        def __init__(self, kernel_size=5, sigma=1.0):
            self.kernel_size = kernel_size
            self.sigma = sigma
    
        def __call__(self, frame):
            return self.apply_gaussian_blur(frame)
    
        def apply_gaussian_blur(self, frame):
            # Convert PIL Image to NumPy array
            frame_np = np.array(frame)
            # Apply Gaussian blur
            frame_np = cv2.GaussianBlur(frame_np, (self.kernel_size, self.kernel_size), self.sigma)
            # Convert NumPy array back to PIL Image
            return Image.fromarray(frame_np)"""""


class BilateralFilter:
    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, frame):
        return self.apply_bilateral_filter(frame)

    def apply_bilateral_filter(self, frame):
        # Convert PIL Image to NumPy array
        frame_np = np.array(frame)

        # Apply bilateral filter
        frame_filtered = cv2.bilateralFilter(frame_np, self.d, self.sigma_color, self.sigma_space)

        # Convert NumPy array to PyTorch tensor
        frame_tensor = torch.from_numpy(frame_filtered).permute(2, 0, 1).float() / 255.0

        return frame_tensor
class Sample_frames_from_list:
    def __init__(self, sample_interval):
        self.sample_interval = sample_interval

    def __call__(self, frame):
        return self.sampled_frames(frames)

    def sampled_frames(self, frames):
        sampled_frames = frames[::self.sample_interval]
        return sampled_frames

def aspect_preserving_resize(img):
    w, h = img.size
    if w > h:
        new_w, new_h = 100, int(100 * h / w)
    else:
        new_w, new_h = int(100 * w / h), 100
    return img.resize((new_w, new_h), Image.LANCZOS)


class Pad:
    def __init__(self, target_height, target_width, padding_value=0):
        self.target_height = target_height
        self.target_width = target_width
        self.padding_value = padding_value

    def __call__(self, frame):
        width, height = frame.size
        delta_w = self.target_width - width
        delta_h = self.target_height - height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_frame = ImageOps.expand(frame, padding, fill=self.padding_value)
        return new_frame

def preprocess_frames(frames, d, sigma_color, sigma_space,sample_interval, target_height, target_width):
    sampled_frames= Sample_frames_from_list(sample_interval=sample_interval).sampled_frames(frames)
    preprocessed_frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        #we need to transform the image into a PIL image, in order to apply some transformations such as resizing
        transforms.Lambda(aspect_preserving_resize),
        Pad(target_height=target_height,target_width=target_width),  # Apply padding to match target size
        BilateralFilter(d=d, sigma_color=sigma_color, sigma_space=sigma_space),
    ])

    for frame in sampled_frames:
        preprocessed_frame = transform(frame)
        preprocessed_frames.append(preprocessed_frame)

    return preprocessed_frames


videos_db = VideosDatabase("/Users/serenatrovalusci/Documents/UNI/video_folder")
video_ids = videos_db.get_ids()

for video_id in video_ids:
    frames = videos_db.read(video_id)
    preprocessed_frames = preprocess_frames(frames, 9, 75, 75, 2, 200, 200)
    # Convert preprocessed frames to NumPy arrays

    preprocessed_frames_np = []
    for preprocessed_frame in preprocessed_frames:
        preprocessed_f = preprocessed_frame.permute(1, 2, 0).numpy()
        preprocessed_f = np.clip(preprocessed_f * 255, 0, 255).astype(np.uint8)
        preprocessed_frames_np.append(preprocessed_f)


""" print(preprocessed_frames)
# Save the preprocessed video
new_video_id = f"{video_id}_preprocessed"
videos_db.save(preprocessed_frames_np, new_video_id, fps=30, size=(640, 640), extension='mp4')"""


"""  # Display the preprocessed video
for frame in preprocessed_frames_np:
    cv2.imshow(f"Video {video_id}", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        
    cv2.destroyAllWindows()"""

show_tracked_video(preprocessed_frames_np)

