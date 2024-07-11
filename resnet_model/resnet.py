from typing import List

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import R2Plus1D_18_Weights


class ResNet2Plus1D(nn.Module):
    """
    ResNet3D model with pretrained weights used for event classification.ù
    Information about objects.
    More information about the used model can be found here:
    https://arxiv.org/abs/1711.11248
    """

    def __init__(self, num_classes=40, num_objects=10, hidden_size=128, unfrozen_layers=4, transforms=None):
        """
        It creates ResNet3D model with pretrained weights used for event classification.
        All the weights are frozen but the last n ones.
        :param num_classes: number of classes to predict.
        :param num_objects: number of objects to combine with the ResNet for classification.
        :param hidden_size: hidden size of the ResNet.
        :param unfrozen_layers: number of the last layers of ResNet to unfreeze for training.
        """
        super(ResNet2Plus1D, self).__init__()
        self.transforms = transforms
        weights = R2Plus1D_18_Weights.DEFAULT
        resnet = models.video.r2plus1d_18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last FC layer
        self.combination_layer = nn.Linear(resnet.fc.in_features + num_objects, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # Freeze all layers except the last `N` layer
        for name, param in self.named_parameters():
            param.requires_grad = False

        # Unfreeze the last `unfreeze_last_n_layers` layers
        layers_to_unfreeze = list(self.named_parameters())[-unfrozen_layers * 2:]
        for name, param in layers_to_unfreeze:
            param.requires_grad = True

    def forward(self, video, objects):
        """
        :param video: input video. A tensor of shape (batch_size, frames, channels - RGB, height, width)
        :param objects: a tensor of shape (batch_size, N) where N is the number of objects and each element in position
            i is a number indicating the number of objects of type i in the video.
        :return: a tensor of shape (batch_size, predictions).
        """
        if len(video.shape) == 4 and len(objects.shape) == 1:
            video = video.unsqueeze(0)
            objects = objects.unsqueeze(0)
        # Swapping dimensions 2 and 3
        video = video.permute(0, 2, 1, 3, 4)
        video = self.features(video)
        video = video.squeeze(4).squeeze(3).squeeze(2)  # B x C x 1 x 1 -> B x C
        combined_input = torch.cat((video, objects), 1)
        x = self.combination_layer(combined_input)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return self.softmax(x)


class ResNet3DClassifier:

    def __init__(self, model: ResNet2Plus1D):
        self.model = model

    def classify(self, video, objects):
        predictions = self.model(video, objects)
        class_id = torch.argmax(predictions, dim=1)
        return class_id


class ReducedResNet3DClassifier(ResNet3DClassifier):

    def __init__(self, model: ResNet2Plus1D, classes_ids: List[int]):
        """
        :param model: The ResNet3D model with pretrained weights used for event classification.
        :param classes_ids: the original classes ids.
            If the model predicts the class "3", the element with index 3 of classes ids is returned-
        """
        super().__init__(model)
        self.classes_ids = classes_ids

    def classify(self, video, objects):
        predictions = self.model(video, objects)
        class_id = torch.argmax(predictions, dim=1)
        return self.classes_ids[class_id]
