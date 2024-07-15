Note: Trained model are already available, so It's not mandatory to execute the training again.

Running the main program:
    0. Navigate to the folder "intelligent-surveillance"
    1. Run: python detector.py [video_path]
Some examples of videos are in the folder 'examples'.
Performances - mostly on action recognition - are low due to a lack of computational power for a good training process.
------------------------------------------------------------------------------------------------------------------------

Dataset processing steps:
    0. Navigate to the folder "intelligent-surveillance"
    1. Download VIRAT dataset "https://viratdata.org/#getting-data".
        The videos from: 'https://data.kitware.com/#collection/611e77a42fa25629b9daceba'
        The annotations from: 'https://gitlab.kitware.com/viratdata/viratannotations'
    2. Put all the videos and annotations in the same folders - respectively - './VIRAT/videos' and './VIRAT/annotations'
    3. Divide the videos in sub-videos and images useful for the models with: python ./preprocess_database.py 1
    4. Process the divided videos and images with: python ./preprocess_database.py 2
    5. Everything will be saved inside the folder ./dataset

Training YOLO model:
    0. Navigate to the folder "intelligent-surveillance"
    1. Run: python YOLO/train.py
    2. Results will be saved in YOLO/runs in a new created folder

Training Res2P1D model:
    0. Navigate to the folder "intelligent-surveillance"
    1. Run: python resnet/train.py
    2. Results will be saved in resnet/models overriding the existing weights

