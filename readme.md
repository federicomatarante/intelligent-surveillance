# ROS1-Planner
## 🇮🇹 Italiano

**Disclaimer**: applicazione fatta per l'esame di _Computer Vision_, nel primo anno della facoltà magistrale di _AI & Robotics_ di _Sapienza Università di Roma_.

Le telecamere di sorveglianza tradizionali hanno un grosso limite: c'è sempre bisogno di qualcuno che stia lì a guardare. E sappiamo tutti come va a finire - dopo un po' la concentrazione cala, ci si distrae, e quando succede qualcosa di importante magari non ce ne accorgiamo in tempo.

L'idea era: "E se potessimo insegnare al computer a guardare al posto nostro?"

## Come funziona

Abbiamo creato un sistema che funziona un po' come il cervello umano quando guarda un video:

1. **Prima identifica gli oggetti** - "Ok, qui c'è una persona, là una macchina"
2. **Li segue nel tempo** - "Quella persona si sta muovendo da sinistra a destra"  
3. **Capisce cosa stanno facendo** - "Ah, quella persona sta correndo, quell'altra sembra che stia litigando"

Il tutto in tempo reale, senza bisogno che nessuno stia lì a guardare.

## La tecnologia

### YOLOv8 per il riconoscimento oggetti
Per riconoscere gli oggetti nei video abbiamo usato YOLOv8, che è tipo il Ferrari dei sistemi di object detection. È velocissimo e abbastanza preciso.

### Object Tracking
Una volta che sappiamo dove sono gli oggetti, bisogna seguirli frame per frame. Ogni oggetto ha la sua "carta d'identità" e possiamo seguirlo mentre si muove nel video.

### R2Plus1D-18 per le azioni
Qui viene la parte più interessante: capire cosa stanno facendo le persone. Abbiamo usato un modello che è specializzato nell'analizzare video, guardando sia nello spazio che nel tempo.

## Dataset

Abbiamo usato il dataset VIRAT:
- 250 ore totali di video di sorveglianza
- 13 tipi di oggetti diversi (persone, auto, ecc.)
- 41 tipi di azioni/eventi diversi


# How to use?
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

