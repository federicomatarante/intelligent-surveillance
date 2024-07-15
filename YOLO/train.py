import os

from ultralytics import YOLO
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    # Verifica se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Carica un modello YOLOv8 pre-addestrato
    model = YOLO(os.path.join(script_dir,'yolov8m.pt'))
    models = os.path.join(script_dir, 'models')

    # Imposta il dispositivo per il modello
    model.to(device)

    # Percorso al file YAML del dataset
    YAML_FILE = os.path.join(script_dir, 'virat_dataset.yaml')

    try:
        # Addestra il modello
        results = model.train(data=YAML_FILE, epochs=100, device=device, project=models)
        print("Training results:", results)

        # Valida il modello
        metrics = model.val()
        print(f"mAP50-95: {metrics.box.map:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Trying to train on CPU...")
        # Prova ad addestrare sulla CPU se si verifica un errore
        results = model.train(data=YAML_FILE, epochs=100, device='cpu', project=models)
        print("Training results:", results)

        # Valida il modello
        metrics = model.val()
        print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == '__main__':
    main()
