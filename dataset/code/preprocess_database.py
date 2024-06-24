import sys
from typing import Dict

from dataset.code.annotations import Annotations, Event, TrackedObject
from dataset.code.annotations_parser import AnnotationsReader
from dataset.code.database import VideosDatabase, ImagesDatabase, AnnotationsDatabase
from dataset.code.dataset_analyzer import DatasetAnalyzer
from dataset.code.video_divider import VideosDivider
from dataset.code.videoprocessor import VideoProcessor

# TODO ideas:
#   1) Do preprocessing of single videos
#   2) Do some weighting for the classes
#   3) track video ID to each track in YOLO

# The folder with all the VIRAT videos.
RAW_VIDEOS_DATABASE = r"C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\videos"
# The folder with the associated annotations to the videos.
RAW_ANNOTATIONS_DATABASE = r"C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\kml_annotations\all"
# The folder that will contain all the events from the VIRAT videos.
EVENT_VIDEOS_DATABASE = r"C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\preprocessed_dataset\events"
# The folder that will contain all the processed videos of events from the VIRAT videos.
PROCESSED_EVENT_VIDEOS_DATABASE = r""r"C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\preprocessed_dataset\processed_events"
# The folder that will contain all the annotations related to the events.
PROCESSED_EVENT_ANNOTATIONS_DATABASE = r"C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\preprocessed_dataset\events_annotations"
# The folder that will contain all the images containing the object tracking from the VIRAT videos.
PROCESSED_TRACKING_IMAGES_DATABASE = r"C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\preprocessed_dataset\tracking"
# The folder that will contain all the annotations related to the object tracking.
PROCESSED_TRACKING_ANNOTATIONS_DATABASE = r"C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\preprocessed_dataset\tracking_annotations"

# The maximum number of images to extract from a video for object tracking.
IMAGES_PER_VIDEO = 10
# The frames to keep before and after an event for video segmentation.
FRAMES_OFFSET = 10
# The offset to add to each direction for the event window when trimming the video.
EVENT_WINDOW_OFFSET = 5
# The minimum number of frames to extract when creating "Empty" events.
MINIMUM_FRAMES = 15
# The minimum size of the video patch to extract "Empty" events
MIN_ZOOM_SIZE = (30, 30)
# The maximum duration of the event window in frames ( VIRAT dataset works in 30 FPS ).
MAX_EVENT_DURATION = 30 * 5


def first_step_processing():
    """
    In this step the VIRAT dataset gets divided in multiple files and annotations.
    Given one video with associated KML annotations from VIRAT:
        - For each event, it creates a short video file and one annotation file describing it.
        - It creates a lot of images for object tracking and annotations related to each image.
    You can modify the hyperparameters above this method to change the behaviour.
    """
    annotations_reader = AnnotationsReader(RAW_ANNOTATIONS_DATABASE)
    videos_divider = VideosDivider(
        videos_folder=RAW_VIDEOS_DATABASE,
        events_folder=EVENT_VIDEOS_DATABASE,
        events_annotations_folder=PROCESSED_EVENT_ANNOTATIONS_DATABASE,
        tracking_folder=PROCESSED_TRACKING_IMAGES_DATABASE,
        tracking_annotations_folder=PROCESSED_TRACKING_ANNOTATIONS_DATABASE,
        images_per_video=IMAGES_PER_VIDEO,
        frames_offset=FRAMES_OFFSET,
        minimum_frames=MINIMUM_FRAMES,
        event_window_offset=EVENT_WINDOW_OFFSET,
        max_event_duration=MAX_EVENT_DURATION,
        min_zoom_size=MIN_ZOOM_SIZE,
    )

    print("Reading raw annotations... ")
    annotations: Dict[str, Annotations] = annotations_reader.read()
    print("Annotations read!")
    print("Dividing the videos in events and tracking images... ")
    videos_divider.divide_videos(annotations, verbose_level=1)
    print("Videos dividing done!")


def second_step_processing():
    """
    In this step each video and image from the "event" and "tracking" folders are processed to improve
    the quality of the dataset.
    """
    event_videos_database = VideosDatabase(EVENT_VIDEOS_DATABASE)
    processed_events_videos_database = VideosDatabase(PROCESSED_EVENT_VIDEOS_DATABASE)
    images_tracking_database = ImagesDatabase(PROCESSED_TRACKING_IMAGES_DATABASE)  # TODO preprocess also these?
    print("Processing the videos... ")
    # TODO maybe add verbose to processor
    processor = VideoProcessor(
        source_database=event_videos_database,
        target_database=processed_events_videos_database,
        d=9,  # Size of the Bilateral filter
        sigma_color=75, sigma_space=75,  # Parameters of the Bilateral filter

    )
    processor.process_videos()


def third_step_processing():
    """
    In this stage simply it will be analyzed the distribution of the labels of the events and tracking
    from the annotations dataset.
    These values can be used to give different weights for each class while training.
    """
    print("Analyzing annotations...\n")
    dataset_analyzer = DatasetAnalyzer(
        videos_annotations_database=AnnotationsDatabase(PROCESSED_EVENT_ANNOTATIONS_DATABASE),
        images_annotations_database=AnnotationsDatabase(PROCESSED_TRACKING_ANNOTATIONS_DATABASE)
    )
    print("Analyzing event tracking...")
    print("\tTotal events: ", dataset_analyzer.events_count)
    print("\tTotal unique labels: ", len(dataset_analyzer.get_event_labels_distribution()))

    event_distribution = dataset_analyzer.get_event_labels_distribution().items()
    print("\tDistribution of label of events: ")
    for i, (label, value) in enumerate(event_distribution):
        label_name = Event.get_label_name(label)
        percentage_value = (value * 100) / dataset_analyzer.events_count
        print(f"\t\t{i} - {label_name} ( ID {label} ) : {value} ({percentage_value:.3f} %)")

    print("\n----------------------\n")
    print("Analyzing tracking...")
    print("\tTotal tracking labels: ", dataset_analyzer.tracked_objects_count)
    print("\tTotal unique labels: ", len(dataset_analyzer.get_tracking_labels_distribution()))
    print("\tDistribution of label of object tracking: ")
    tracking_distribution = dataset_analyzer.get_tracking_labels_distribution().items()
    for i, (label, value) in enumerate(tracking_distribution):
        label_name = TrackedObject.get_label_literal(label)
        percentage_value = (value * 100) / dataset_analyzer.tracked_objects_count
        print(f"\t\t{i} - {label_name} ( ID {label} ) : {value} ({percentage_value:.3f} %)")


def main():
    """
    Main function for preprocessing and analyzing the dataset.
    Usage: python preprocess_database.py [1|2|3]
    Careful! Set right hyperparameters at the top of this file before running this script.
    """
    usage = ("python3 preprocess_database.py [1|2|3]\n"
             "Where:\n"
             "\t 1: First Step Preprocessing ( Videos from VIRAT dataset are cut in useful chuncks or images )\n"
             "\t 2: Second Step Preprocessing ( The chuncks of videos and images are preprocessed  )\n"
             "\t 3: Third Step Preprocessing ( The labels of events and object tracking are analyzed )\n")
    if len(sys.argv) != 2 or sys.argv[1] not in ["1", "2", "3"]:
        print(usage)
        return 1

    if RAW_VIDEOS_DATABASE == "" or RAW_ANNOTATIONS_DATABASE == "" or EVENT_VIDEOS_DATABASE == "" or PROCESSED_TRACKING_ANNOTATIONS_DATABASE == "":
        print("Set the hyperparameters at the top of this file before running the script!")
        return 1

    if sys.argv[1] == "1":
        first_step_processing()
    elif sys.argv[1] == "2":
        second_step_processing()
    else:
        third_step_processing()


if __name__ == '__main__':
    main()
