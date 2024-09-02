import os
import cv2
import timm
import torch
import numpy as np
from torch import nn
import torchvision.models as models
from albumentations import (Compose, Normalize, Resize, HorizontalFlip)
# from PIL import Image

# import random
# import pandas as pd
# import sys


class model:
    def __init__(self):
        self.checkpoint1 = "ResNet50d_best_fold_1.pt"
        self.checkpoints1 = ['ResNet50d_best_fold_1.pt', 'ResNet50d_best_fold_2.pt',
                             'ResNet50d_best_fold_3.pt', 'ResNet50d_best_fold_4.pt']

        self.checkpoint2 = "ResNet34d_best_fold_1.pt"
        self.checkpoints2 = ['ResNet34d_best_fold_1.pt', 'ResNet34d_best_fold_2.pt',
                             'ResNet34d_best_fold_3.pt', 'ResNet34d_best_fold_4.pt']
        self.checkpoints3 = ['ResNet34dp_best_fold_1.pt', 'ResNet34dp_best_fold_2.pt',
                             'ResNet34dp_best_fold_3.pt', 'ResNet34dp_best_fold_4.pt']
        self.model_name = '50d'
        self.device = torch.device("cpu")
        self.tta = True
        self.multi_folds = True
        self.img_size = 512
        self.multi_models = True
        self.use_models = ['ResNet50d', 'ResNet34dp']

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        if not self.multi_folds:
            if self.model_name == '50d':
                self.model1 = ResNet50d(num_classes=2)
                checkpoint_path = os.path.join(dir_path, self.checkpoint1)
            elif self.model_name == '34d':
                self.model1 = ResNet34d(num_classes=2)
                checkpoint_path = os.path.join(dir_path, self.checkpoint2)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if any(key.startswith("module.") for key in state_dict.keys()):
                new_state_dict = {key[7:]: value for key, value in state_dict.items()}
            else:
                new_state_dict = state_dict
            self.model1.load_state_dict(new_state_dict)
            self.model1.to(self.device)
            self.model1.eval()
        else:
            self.models1 = []
            if not self.multi_models:
                if self.model_name == '50d':
                    tmp_mode = ResNet50d(num_classes=2)
                    checkpoints = self.checkpoints1
                elif self.model_name == '34d':
                    tmp_mode = ResNet34d(num_classes=2)
                    checkpoints = self.checkpoints2
                for point in checkpoints:
                    checkpoint_path = os.path.join(dir_path, point)
                    state_dict = torch.load(checkpoint_path, map_location=self.device)
                    if any(key.startswith("module.") for key in state_dict.keys()):
                        new_state_dict = {key[7:]: value for key, value in state_dict.items()}
                    else:
                        new_state_dict = state_dict
                    tmp_mode.load_state_dict(new_state_dict)
                    tmp_mode.to(self.device)
                    tmp_mode.eval()
                    self.models1.append(tmp_mode)
            else:
                for model_name in self.use_models:
                    if model_name == 'ResNet50d':
                        tmp_mode = ResNet50d(num_classes=2)
                        checkpoints = self.checkpoints1
                    elif model_name == 'ResNet34d':
                        tmp_mode = ResNet34d(num_classes=2)
                        checkpoints = self.checkpoints2
                    elif model_name == 'ResNet34dp':
                        tmp_mode = ResNet34d(num_classes=2)
                        checkpoints = self.checkpoints3
                    for point in checkpoints:
                        checkpoint_path = os.path.join(dir_path, point)
                        state_dict = torch.load(checkpoint_path, map_location=self.device)
                        if any(key.startswith("module.") for key in state_dict.keys()):
                            new_state_dict = {key[7:]: value for key, value in state_dict.items()}
                        else:
                            new_state_dict = state_dict
                        tmp_mode.load_state_dict(new_state_dict)
                        tmp_mode.to(self.device)
                        tmp_mode.eval()
                        self.models1.append(tmp_mode)

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).

        !! Note that the order of the three channels of the input_image read by cv2.imread is BGR. This is the way we use to read the image.
        !! If you use Image.open() from PIL in your training process, the order of the three channels will be RGB. Please pay attention to this difference.

        :param input_image: the input image to the model.
        :return: a float value indicating the probability of class 1.
        """
        image = cv2.resize(input_image, (self.img_size, self.img_size))
        with torch.no_grad():
            if self.tta:
                predictions = []
                tta_transforms = self.test_time_aug()
                for transform in tta_transforms:
                    augmented = transform(image=np.array(image))
                    augmented_image = augmented['image']
                    augmented_image = torch.from_numpy(augmented_image).permute(2, 0, 1).unsqueeze(0)
                    augmented_image = augmented_image.to(self.device, torch.float)
                    if self.multi_folds:
                        output = self.predict_models(augmented_image, self.models1)
                    else:
                        output = self.model1(augmented_image)
                    predictions.append(output)
                output = torch.mean(torch.stack(predictions), dim=0)
            else:
                tta_transforms = self.test_time_aug()
                transform = tta_transforms[0]
                augmented = transform(image=image)
                augmented_image = augmented['image']
                augmented_image = torch.from_numpy(augmented_image).permute(2, 0, 1).unsqueeze(0)
                augmented_image = augmented_image.to(self.device, torch.float)
                if self.multi_folds:
                    output = self.predict_models(augmented_image, self.models1)
                else:
                    output = self.model1(augmented_image)
            prob = torch.softmax(output, dim=1).squeeze(0)

        class_1_prob = prob[1]
        class_1_prob = class_1_prob.detach().cpu()
        return float(class_1_prob)


    def predict_models(self, image, my_models):
        predictions = []
        for my_model in my_models:
            output = my_model(image)
            predictions.append(output)
        return torch.mean(torch.stack(predictions), dim=0)


    def test_time_aug(self):
        return [
            Compose([
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]),
            Compose([
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                HorizontalFlip(p=1.0)
            ])]


class ResNet34d(nn.Module):
    def __init__(self, model_name='resnet34d', pretrained=False, num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50d(nn.Module):
    def __init__(self, model_name='resnet50d', pretrained=False, num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False, num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class SeResNext(nn.Module):
    def __init__(self, model_name='legacy_seresnext26_32x4d', pretrained=False, num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Linear(num_features, num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x

