import json
import os
import subprocess
import time
from typing import Tuple, Any

import torch
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

from data.annotations import tracked_object_labels
from data.database import AnnotationsDatabase
from data.dataset_analyzer import DatasetAnalyzer
from data.datasets import ActionRecognitionDataset, ReducedActionRecognitionDataset, VideoCollater
from preprocess_database import PROCESSED_EVENT_VIDEOS_DATABASE, PROCESSED_EVENT_ANNOTATIONS_DATABASE
from resnet.resnet import ResNet2Plus1D


def get_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    return int(result.strip())


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, dataset, batch_size=10, max_frames=150, class_names=None, verbose=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=VideoCollater(max_frames))

    all_preds = []
    all_true_vals = []
    model.to(device)
    with torch.no_grad():
        for i, (frames, objects, labels) in enumerate(data_loader):
            frames = frames.to(device)
            objects = objects.to(device)
            labels = labels.to(device)
            outputs = model(frames, objects)
            _, preds = torch.max(outputs, 1)
            _, true_vals = torch.max(labels, 1)

            all_preds.extend(preds.cpu().numpy())
            all_true_vals.extend(true_vals.cpu().numpy())

            del frames, objects, labels, outputs
            torch.cuda.empty_cache()

    all_preds = np.array(all_preds)
    all_true_vals = np.array(all_true_vals)

    accuracy = accuracy_score(all_true_vals, all_preds)
    f1 = f1_score(all_true_vals, all_preds, average='weighted')
    precision = precision_score(all_true_vals, all_preds, average='weighted')
    recall = recall_score(all_true_vals, all_preds, average='weighted')

    if verbose:
        print("Accuracy: {:.4f}%".format(accuracy * 100))
        print("F1 Score: {:.4f}".format(f1))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))

    if verbose > 1:
        cm = confusion_matrix(all_true_vals, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if class_names:
            plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
            plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

    return accuracy


def save_model(directory, model_id, model, training_data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, "model_" + model_id + ".pt")
    training_data_path = os.path.join(directory, "training_data_" + model_id + ".json")

    torch.save(model.state_dict(), str(model_path))
    with open(training_data_path, 'w') as f:
        json.dump(training_data, f)


def train(
        model: nn.Module,
        dataset: ActionRecognitionDataset,
        num_epochs: int,
        batch_size: int,
        val_batch_size: int,
        sub_dataset_size: int,
        directory: str,
        save_every_epochs: int,
        test_dataset: ReducedActionRecognitionDataset,
        model_name: str,
        lr: float = 1e-3,
        split_ratio: float = 0.8,
        classes_proportions: Tuple[float, ...] = None,
        max_frames=150,
        verbose=0
) -> tuple[Module, dict[str | Any, int | float | str | list[float] | tuple[float, ...] | None | Any]]:
    """
    Train an action recognition model using PyTorch.

    Args:
        model (nn.Module): The action recognition model to be trained, an instance of a class inheriting from nn.Module.
        dataset (ActionRecognitionDataset): The dataset, an instance of the ActionRecognitionDataset class. It will be
            split into training and validation sets.
        num_epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size to use during training.
        lr (float): The learning rate for the optimizer.
        split_ratio: the proportion of the dataset to use for training. Must be between 0 and 1.
        classes_proportions: the proportions of the classes to use for training to give weights to classes.
        sub_dataset_size: size of the subset of the datasett to sample.
        val_batch_size: the batch size of the validation set.
        max_frames: the maximum number of frames to use during training per video.
        verbose: 0 for silent, 1 for verbose and 2 for debugging.
        directory: the directory in which to save the model.
        save_every_epochs: every numer of epoch in which to save the model.
        test_dataset: the dataset to use for testing.
        model_name: the name of the model.
    Returns:
        nn.Module: The trained action recognition model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu' and verbose:
        print("Careful! You're doing the training on CPU only, It's highly reccomended to use a CUDA device.")
    # Create data loaders
    if split_ratio < 0 or split_ratio > 1:
        raise ValueError("Split ratio must be between 0 and 1.")
    train_dataset, val_dataset = random_split(dataset, [split_ratio, 1 - split_ratio])

    # Initialize loss function and optimizer
    if classes_proportions:
        sum_proportions = sum(classes_proportions)
        classes_proportions = [class_proportion / sum_proportions for class_proportion in classes_proportions]
        weights = [1 - class_proportion for class_proportion in
                   classes_proportions]
        weights = np.array(weights, dtype=float)
        weights = np.log1p(weights)
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        max_weight = 1
        min_weight = 0.7
        weights = weights * (max_weight - min_weight) + min_weight
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if verbose:
        print("Training settings: ")
        print("\tDevice: {}".format(device))
        print("\tSize of subsets: {}".format(sub_dataset_size))
        print("\tEpochs: {}".format(num_epochs))
        print("\tBatch size: {}".format(batch_size))
        print("\tLearning rate: {}".format(lr))
        print("\tTraining set size: {}".format(len(dataset)))
        print("\tBatches per epoch: {}".format(int(sub_dataset_size * split_ratio / batch_size)))
        print("\tOptimizer used: {}".format(optimizer.__class__.__name__))
        print("\tLoss used: {}".format(criterion.__class__.__name__))
        print("\tTraining set size: {}".format(len(train_dataset)))
        print("\tValidation set size: {}".format(len(val_dataset)))
        if classes_proportions:
            print("\tClasses proportions: {}".format(
                [f'{proportion:.4f}' for proportion in list(classes_proportions)]))
            print("\tClass weights: {}".format(
                [f'{weight:.4f}' for weight in list(weights)]))

    model = model.to(device)
    criterion = criterion.to(device)
    if verbose:
        print("Training model...")
    # Training data
    train_losses = []
    val_losses = []
    accuracies = []
    times = []

    # Training loop
    for epoch in range(num_epochs):
        train_indices = np.random.choice(len(train_dataset), size=int(sub_dataset_size * split_ratio), replace=False)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=VideoCollater(max_frames),
                                  num_workers=4, sampler=train_sampler)
        val_indices = np.random.choice(len(val_dataset), size=int(sub_dataset_size * (1 - split_ratio)), replace=False)
        train_sampler = SubsetRandomSampler(val_indices)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=VideoCollater(max_frames),
                                num_workers=4, sampler=train_sampler)
        if verbose:
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
        start_time = time.time()
        # Training
        model.train()
        train_loss = 0.0
        batches_num = len(train_loader)
        for i, (frames, objects, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            frames = frames.to(device)
            objects = objects.to(device)
            labels = labels.to(device)
            if verbose > 1:
                print("Frames shape: ", frames.shape)
            outputs = model(frames, objects)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if verbose > 1:
                print(f"\tBatch {i + 1}/{batches_num}: GPU memory used: {get_gpu_memory()} MB")

        if verbose > 1:
            print("Training ended, evaluating...")
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        batches_num = len(val_loader)
        with torch.no_grad():
            for i, (frames, objects, labels) in enumerate(val_loader):
                frames = frames.to(device)
                if verbose > 1:
                    print("Frames shape: ", frames.shape)
                objects = objects.to(device)
                labels = labels.to(device)
                outputs = model(frames, objects)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                _, true_vals = torch.max(labels, 1)
                val_acc += (preds == true_vals).sum().item()
                del frames, objects, labels, outputs
                torch.cuda.empty_cache()
                if verbose > 1:
                    print(f"\tBatch {i + 1}/{batches_num}: GPU memory used: {get_gpu_memory()} MB")

        # Print training and validation metrics
        end_time = time.time()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_dataset)
        time_passed = end_time - start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(val_acc)
        times.append(time_passed)
        if verbose:
            print(
                f'\tTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_acc * 100:.4f}%, Time for training: {time_passed:.4f}')
            print("-" * 40)
        if epoch % save_every_epochs == 0:
            accuracy = evaluate_model(model, test_dataset, batch_size, max_frames)
            train_configuration = {
                'epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'optimizer': optimizer.__class__.__name__,
                'criterion': criterion.__class__.__name__,
                'sub_dataset_size': sub_dataset_size,
                'split_ratio': split_ratio,
                'weights': weights.tolist(),
                'classes_proportions': classes_proportions,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': accuracies,
                'times': times,
                'test_accuracy': accuracy
            }
            save_model(directory, model_name + "-" + str(epoch), model, train_configuration)


def main():
    VIDEOS_DIR = PROCESSED_EVENT_VIDEOS_DATABASE
    ANNOTATIONS_DIR = PROCESSED_EVENT_ANNOTATIONS_DATABASE

    model = ResNet2Plus1D(
        num_classes=12,
        num_objects=len(tracked_object_labels),
        hidden_size=32,
        unfrozen_layers=6
    )

    # New IDS will be respectively: [0,1,2,3,...]
    reduced_classes = [0, 9, 26, 38, 37, 31, 35, 15, 19, 8, 20, 12]

    dataset = ReducedActionRecognitionDataset(
        videos_dir=VIDEOS_DIR,
        annotations_dir=ANNOTATIONS_DIR,
        classes_to_keep=reduced_classes
    )

    dataset_analyzer = DatasetAnalyzer(videos_annotations_database=AnnotationsDatabase(ANNOTATIONS_DIR),
                                       images_annotations_database=None)
    classes_proportions = tuple(dataset_analyzer.get_event_labels_distribution().values())[0:12]
    train_set, test_set = random_split(dataset, [0.8, 0.2])
    print("Train Dataset size: ", len(train_set))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    train(model=model,
          dataset=dataset,
          num_epochs=30 * 6,
          batch_size=2,
          classes_proportions=classes_proportions,
          verbose=1,
          sub_dataset_size=2000,
          val_batch_size=2,
          directory=os.path.join(script_dir, "models"),
          save_every_epochs=120,
          test_dataset=test_set,
          model_name="ResNet2P1",
          )


if __name__ == '__main__':
    main()
