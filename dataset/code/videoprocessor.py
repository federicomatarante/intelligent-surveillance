import numpy as np
from dataset.code.database import VideosDatabase
from dataset.code.preprocess import preprocess_frames

class VideoProcessor:
    def __init__(self, database_path, d, sigma_color, sigma_space,sample_interval, target_height, target_width):
        self.videos_db= VideosDatabase(database_path)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.sample_interval = sample_interval
        self.target_height = target_height
        self.target_width = target_width

    def process_video(self):
        video_ids = self.videos_db.get_ids()

        for video_id in video_ids:
            frames = self.videos_db.read(video_id)
            preprocessed_frames = self.preprocess_frames(frames, 9, 75, 75)
            preprocessed_frames_np = self.convert_to_numpy(preprocessed_frames)
            self.videos_db.save(preprocessed_frames_np, video_id, 15, (200, 200), 'mp4')

    def preprocess_frames(self, frames):
        # Implement your preprocess_frames function here
        # This is a placeholder and should be replaced with actual implementation
        return preprocess_frames(frames, self.d, self.sigma_color, self.sigma_space, self.sample_interval, self.target_height, self.target_width)

    def convert_to_numpy(self, preprocessed_frames):
        preprocessed_frames_np = []
        for preprocessed_frame in preprocessed_frames:
            preprocessed_f = preprocessed_frame.permute(1, 2, 0).numpy()
            preprocessed_f = np.clip(preprocessed_f * 255, 0, 255).astype(np.uint8)
            preprocessed_frames_np.append(preprocessed_f)
        return preprocessed_frames_np




