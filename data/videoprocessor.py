import numpy as np
from data.database import VideosDatabase
from data.preprocess import preprocess_frames


class VideoProcessor:
    def __init__(self, source_database: VideosDatabase, target_database: VideosDatabase, d=9, sigma_color=75, sigma_space=75,
                 sample_interval=2, target_height=100, target_width=100, max_size=100, extension='mp4', fps=30):
        """
        Preprocesses all the videos from the database.
        The filters it applies for each video:
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
        :param sample_interval: the sample interval of the frames, so how many frames to skip for sampling.
        :param target_height: the max height of an image should have after the padding.
        :param target_width: the max width of an image should have after the padding.
        :param max_size: The maximum size ( between height and width ) of the image after downsampling.
        :param extension: The extension of the created videos.
        :param fps: FPS of the videos in the source database.
        """
        self.extension = extension
        self.source_db = source_database
        self.target_db = target_database
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.sample_interval = sample_interval
        self.target_height = target_height
        self.target_width = target_width
        self.max_size = max_size
        self.fps = fps

    def process_videos(self):
        """
        Preprocesses all the videos from the source database and saves them to the target database.
        """
        video_ids = self.source_db.get_ids()

        for video_id in video_ids:
            frames = self.source_db.read(video_id)
            preprocessed_frames = self._preprocess_frames(frames)
            preprocessed_frames_np = self._convert_to_numpy(preprocessed_frames)
            self.target_db.save(preprocessed_frames_np, video_id, int(self.fps / self.sample_interval),
                                (self.target_height, self.target_width), self.extension)

    def _preprocess_frames(self, frames):
        """
        It preprocesses the frames according to the given parameters.
        :param frames: the list of frames to be preprocessed.
        :return: the preprocessed list of frames.
        """
        return preprocess_frames(frames, self.d, self.sigma_color, self.sigma_space, self.sample_interval,
                                 self.target_height, self.target_width, self.max_size)

    @staticmethod
    def _convert_to_numpy(preprocessed_frames):
        """
        Converts a  list of tensors frames into a numpy array.
        """
        preprocessed_frames_np = []
        for preprocessed_frame in preprocessed_frames:
            preprocessed_frames_np.append(np.array(preprocessed_frame))
        return preprocessed_frames_np

