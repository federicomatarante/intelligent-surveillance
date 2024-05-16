import sys
from cmath import sqrt

import cv2
# pip install opencv-python
from torch import tensor


def compute_MoSIFT(frame1: tensor, frame2: tensor,
                   octaves=4, levels=5, k=sqrt(2), sigma=1.6,
                   bins=36, patch_size=16) -> tensor:
    """
    Computes the MoSIFT descriptor using a Gaussian kernel.
    Parameters:
    :param frame1: First frame of the video.
    :param frame2: Second frame of the video.
    :param octaves: Number of downsampled images.
    :param levels: Number of levels for each of the octaves.
    :param sigma: Standard deviation of the Gaussian kernel.
    :param k: multiplying factor of the Gaussian kernel in each level.
    :param bins: Number of bins of the Histogram of Gradients ( HoG ) for the SIFT keypoints.
    :param patch_size: Size of the patches used to compute the HoG for the SIFT keypoints.

    :return: a tensor of the MoSIFT descriptors. The shape of the tensor is (N,256), where
    N is the number of keypoints and 256 is the size of ad descriptor.
    """
    # Load the image
    image = cv2.imread(r'C:\Users\feder\PycharmProjects\intelligent-surveillance\cinghiale.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints = sift.detect(gray, None)

    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    cv2.imshow('Image with SIFT Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def unpackSIFTOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave & 0xFF
    layer = (_octave >> 8) & 0xFF
    if octave >= 128:
        octave |= -128
    if octave >= 0:
        scale = float(1 / (1 << octave))
    else:
        scale = float(1 << -octave)
    return (octave, layer, scale)


image = cv2.imread(
    r'C:\Users\feder\OneDrive\Desktop\english-dog-breeds-4788340-hero-14a64cf053ca40f78e5bd078b052d97f.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints = sift.detect(gray)

for i in range(len(keypoints)):
    octave, layer, scale = unpackSIFTOctave(keypoints[i])
    print(
        f"Position: {keypoints[i].pt}, Angle: {keypoints[i].angle} "
        f"Octave: {octave} Layer: {layer} Scale: {scale} "
        f"Response: {keypoints[i].response}")

image_with_keypoints = cv2.drawKeypoints(gray, keypoints, image)
cv2.imshow('Image with SIFT Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
