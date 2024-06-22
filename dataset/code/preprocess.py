from typing import List

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps


class BilateralFilter:
    """
    A class for performing bilateral filtering of an image.
    """

    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        """

        :param d: the radius of the bilateral filter.
        :param sigma_color: This parameter controls the filter's behavior in the color domain. It defines how much
            influence the difference in color or intensity between pixels has on the filtering process.
            - A larger sigmaColor value means that pixels with more different colors/intensities will be mixed together,
            resulting in more smoothing and noise reduction.
            - A smaller sigmaColor value means only very similar colors will be mixed, preserving more color edges.
            Typical values are between 75 and 100.
        :param sigma_space: This parameter controls the filter's behavior in the spatial domain. It
            determines how much influence the distance between pixels has on the filtering process.
            - A larger sigmaSpace value means that pixels farther apart will influence each other, resulting in larger areas being smoothed.
            - A smaller sigmaSpace value restricts the effect to closer pixels, leading to more localized smoothing.
            Typical values are between 75 and 100.
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, frame):
        return self.apply_bilateral_filter(frame)

    def apply_bilateral_filter(self, frame):
        frame_np = np.array(frame)
        frame_filtered = cv2.bilateralFilter(frame_np, self.d, self.sigma_color, self.sigma_space)
        frame_filtered = Image.fromarray(frame_filtered)
        return frame_filtered


class FramesSampler:
    """
    A class to reduce the frame rate of a video.
    """

    def __init__(self, sample_interval):
        """
        :param sample_interval: the sample interval of the frames, so how many frames to skip for sampling.
        """
        self.sample_interval = sample_interval

    def __call__(self, frames):
        return self.sampled_frames(frames)

    def sampled_frames(self, frames):
        sampled_frames = frames[::self.sample_interval]
        return sampled_frames


class RatioPreservingResizer:
    """
    Resizes an image keeping the ratio between the width and height.
    For example if (H,W)=(200,400) and the max_value is 100, the resized image will be of size (50,100).
    """

    def __init__(self, max_size: int):
        """
        :param max_size: the minimum value to keep the aspect ratio between height and width.
        """
        self.max_size = max_size

    def __call__(self, frame):
        return self.aspect_preserving_resize(frame)

    def aspect_preserving_resize(self, img):
        """
        :param img: the image to be resized.
        :return: the resized image.
        """
        w, h = img.size
        if w > h:
            new_w, new_h = self.max_size, int(self.max_size * h / w)
        else:
            new_w, new_h = int(self.max_size * w / h), self.max_size
        return img.resize((new_w, new_h), Image.LANCZOS)


class Padder:
    """
    A class to pad an image.
    """

    def __init__(self, target_height, target_width, padding_value=0):
        """

        :param target_height: the max height of an image should have after the padding.
        :param target_width: the max width of an image should have after the padding.
        :param padding_value: the value to be used for the padding pixels.
        """
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


def preprocess_frames(frames: List[np.ndarray], d: int, sigma_color: int, sigma_space: int, sample_interval: int,
                      target_height: int, target_width: int, max_size: int):
    """
    Preprocesses a list of frames to make them suitable for training.
    The filters it applies are:
     - Frame sampling: it decreases the frame rate of the video.
     - Downsample: it downsamples the frames preserving their aspect ratio.
     - Bilateral filtering: it applies bilateral filtering on the frames making them more smooth and preserving edges.
     - Pad: it pads the frames according to the given parameters.
    :param frames: the list of frames to be preprocessed.
    :param d: the radius of the bilateral filter.
    :param sigma_color: This parameter controls the Bilateral filter's behavior in the color domain.
        See 'BilateralFilter' for more details.
    :param sigma_space: This parameter controls the Bilateral filter's behavior in the spatial domain.
        See 'BilateralFilter' for more details.
    :param sample_interval: the sample interval of the frames, so how many frames to skip for sampling.
    :param target_height: the max height of an image should have after the padding.
    :param target_width: the max width of an image should have after the padding.
    :param max_size: The maximum size ( between height and width ) of the image after downsampling.
    :return: the preprocessed list of frames.
    """
    sampled_frames = FramesSampler(sample_interval=sample_interval).sampled_frames(frames)
    preprocessed_frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        RatioPreservingResizer(max_size=max_size),
        BilateralFilter(d=d, sigma_color=sigma_color, sigma_space=sigma_space),
        Padder(target_height=target_height, target_width=target_width),
    ])

    for frame in sampled_frames:
        preprocessed_frame = transform(frame)
        preprocessed_frames.append(preprocessed_frame)

    return preprocessed_frames
