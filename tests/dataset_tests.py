import os
import unittest

from torch.utils.data import Dataset, DataLoader

from annotations import Annotations
from dataset import _DatabaseExtractor, ViratDataset


class DatabaseExtractorTests(unittest.TestCase):
    # Set these parameters to execute your tests
    VIDEOS_DIR = r"test_dataset/videos"
    ANNOTATIONS_DIR = r"test_dataset/annotations"

    def setUp(self):
        self.database_extractor = _DatabaseExtractor(videos_dir=self.VIDEOS_DIR, annotations_dir=self.ANNOTATIONS_DIR)

    def test_reading_annotations(self):
        annotations = self.database_extractor.get_annotations()
        expected_keys = ["events", "objects", "mapping"]
        for video_name, annotation_data in annotations.items():
            for key in expected_keys:
                self.assertIn(key, annotation_data.keys(), f"Key '{key}' missing for the video '{video_name}'")

        event_keys = ["eventID", "eventType", "duration", "startFrame", "endFrame", "currentFrame",
                      "bbox_lefttop_x", "bbox_lefttop_y", "bbox_width", "bbox_height"]
        objects_keys = ["objectID", "objectDuration", "currentFrame", "bbox_lefttop_x", "bbox_lefttop_y",
                        "bbox_width", "bbox_height", "objectType"]
        mapping_keys = ["eventID", "eventType", "event_duration", "startFrame", "endFrame", "number_of_obj",
                        "relationsToObject"]
        for video_name, annotation_data in annotations.items():
            for annotation_type, annotation_details in annotation_data.items():
                for annotation_detail in annotation_details:

                    if annotation_type == "events":
                        keys = event_keys
                    elif annotation_type == "objects":
                        keys = objects_keys
                    else:
                        keys = mapping_keys

                    for expected_key in keys:
                        self.assertIn(expected_key, annotation_detail, f"Key '{expected_key}' missing "
                                                                       f"fo the annotation type '{annotation_type}' "
                                                                       f"for the video '{video_name}'")
                        if expected_key == 'relationsToObject':

                            self.assertIsInstance(annotation_detail[expected_key], list,
                                                  f"The value associated with the key '{expected_key}' is not an integer for "
                                                  f"the annotation type '{annotation_type}' for the video '{video_name}'")
                            assert all(isinstance(x, int) for x in
                                       annotation_detail[expected_key]), "Not all elements in the list are integers"

                        else:
                            self.assertIsInstance(annotation_detail[expected_key], int,
                                                  f"The value associated with the key '{expected_key}' is not an integer for "
                                                  f"the annotation type '{annotation_type}' for the video '{video_name}'")

    def test_video_paths(self):
        video_paths = self.database_extractor.get_video_paths()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg']
        for video_path in video_paths:
            extension = os.path.splitext(video_path)[1]
            self.assertIn(extension, video_extensions, msg=f"Wrong video path. Found the extension '{extension}'")
            found_dir = os.path.split(video_path)[0]
            expected_dir = self.database_extractor.videos_dir
            assert found_dir in expected_dir, f"Wrong video path for video '{video_path}'\n Expected: {expected_dir}. Found: {found_dir}"

    def test_load_videos(self):
        video_paths = self.database_extractor.get_video_paths()
        video_names = [os.path.basename(video_path) for video_path in video_paths]
        for video_name in video_names:
            video = self.database_extractor.load_video(video_name)
            tensor_shape = video.shape
            # Expected shape:  (num_frames, height, width, channels)
            self.assertEqual(len(tensor_shape), 4,
                             msg=f"Wrong shape for video '{video_name}'. Expected: (num_frames, height, width, "
                                 f"channels). Found: {tensor_shape}")

    def test_dataset(self):
        dataset = ViratDataset(self.database_extractor.videos_dir,
                               self.database_extractor.annotations_dir)

        videos_count = len(self.database_extractor.get_video_paths())
        self.assertEqual(videos_count, len(dataset),
                         f"The length of the dataset is incorrect. It should be {videos_count} but it's {len(dataset)}")
        for record in dataset:
            self.assertIsInstance(record, dict,
                                  f"The return type of the database is incorrect. It should be a dict. It should be 'dict' but it's {type(record)}")
            self.assertIn("frames", record)
            self.assertIn("annotations", record)
            frames = record["frames"]
            tensor_shape = frames.shape
            # Expected shape:  (num_frames, height, width, channels)
            self.assertEqual(len(tensor_shape), 4,
                             msg=f"Wrong shape for video. Expected: (num_frames, height, width, "
                                 f"channels). Found: {tensor_shape}")
            annotations = record["annotations"]
            self.assertIsInstance(annotations, Annotations)

    def test_data_loader(self):
        dataset = ViratDataset(self.database_extractor.videos_dir,
                               self.database_extractor.annotations_dir)
        batch_sizes = [1, 2, 0]
        for batch_size in batch_sizes:
            print("Batch_size: ", batch_size)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for video, annotations in data_loader:
                print("video: ", video)
                print("annotations: ", annotations)
