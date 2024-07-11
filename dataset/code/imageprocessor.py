from dataset.code.database import ImagesDatabase
from dataset.code.preprocess import preprocess_frame
import numpy as np


class ImageProcessor:
    def __init__(self, source_database: ImagesDatabase, target_database: ImagesDatabase, d=9, sigma_color=75,
                 sigma_space=75, target_height=100, target_width=100, max_size=100, extension='jpeg'):
        """
            Preprocesses all the videos from the database.
            The filters it applies for each image:
             - Frame sampling: it decreases the frame rate of the video.
             - Downsample: it downsamples the frames preserving their aspect ratio.
             - Bilateral filtering: it applies bilateral filtering on the frames making them more smooth and preserving edges.
             - Pad: it pads the frames according to the given parameters.
            :param source_database: the database where the videos to be processed are located.
            :param target_database:  the database where the processed videos will be located.
            :param d: the radius of the bilateral filter.
            :param sigma_color: This parameter controls the Bilateral filter's behavior in the color domain.
                See 'BilateralFilter' for more details.
            :param sigma_space: This parameter controls the Bilateral filter's behavior in the spatial domain.
                See 'BilateralFilter' for more details.
            :param target_height: the max height of an image should have after the padding.
            :param target_width: the max width of an image should have after the padding.
            :param max_size: The maximum size ( between height and width ) of the image after downsampling.
            :param extension: The extension of the created videos.
            """
        self.extension = extension
        self.source_db = source_database
        self.target_db = target_database
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.target_height = target_height
        self.target_width = target_width
        self.max_size = max_size

    def preprocess_images(self):
        """
                    Preprocesses all the images from the source database and saves them to the target database.
            """
        image_ids = self.source_db.get_ids()

        for image_id in image_ids:
            frame = self.source_db.read(image_id)
            preprocessed_frame = self._preprocess_frame(frame)
            preprocessed_frame_np = np.array(preprocessed_frame)
            self.target_db.save(image_id, preprocessed_frame_np)

    def _preprocess_frame(self, frame):
        """
            It preprocesses the frames according to the given parameters.
            :param frame: the frame to be preprocessed.
            :return: the preprocessed list of frames.
            """
        return preprocess_frame(frame, self.d, self.sigma_color, self.sigma_space,
                                self.target_height, self.target_width, self.max_size)
