import os
import cv2
import timm
import torch
import random
import logging
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
import argparse
import pandas as pd
from torch import nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop,
    HorizontalFlip, VerticalFlip, CenterCrop, RandomBrightnessContrast,
    Rotate, ShiftScaleRotate, CoarseDropout, Transpose, MixUp)
from van import van_b1, van_b2
from cbam import ChannelAttention, SpatialAttention


class Model:
    def __init__(self):
        self.checkpoint1 = "ResNet50d_best_fold_1.pt"
        self.checkpoints1 = ['ResNet50d_best_fold_1.pt', 'ResNet50d_best_fold_2.pt',
                             'ResNet50d_best_fold_3.pt', 'ResNet50d_best_fold_4.pt']

        self.checkpoint2 = "ResNet34d_best_fold_1.pt"
        self.checkpoints2 = ['ResNet34d_best_fold_1.pt', 'ResNet34d_best_fold_2.pt',
                             'ResNet34d_best_fold_3.pt', 'ResNet34d_best_fold_4.pt']
        self.checkpoint3 = "GhostNet_best_fold_1.pt"
        self.checkpoints3 = ['GhostNet_best_fold_1.pt', 'GhostNet_best_fold_2.pt',
                             'GhostNet_best_fold_3.pt', 'GhostNet_best_fold_4.pt']
        self.checkpoint4 = "MobileNet_best_fold_1.pt"
        self.checkpoints4 = ['MobileNet_best_fold_1.pt', 'MobileNet_best_fold_2.pt',
                             'MobileNet_best_fold_3.pt', 'MobileNet_best_fold_4.pt']
        self.checkpoints5 = ['vanb1_best_fold_1.pt', 'vanb1_best_fold_2.pt',
                             'vanb1_best_fold_3.pt', 'vanb1_best_fold_4.pt']
        self.checkpoints6 = ['vanb2_best_fold_1.pt', 'vanb2_best_fold_2.pt',
                             'vanb2_best_fold_3.pt', 'vanb2_best_fold_4.pt']
        self.checkpoints7 = ['ResNet34d_CBAM_best_fold_1.pt', 'ResNet34d_CBAM_best_fold_2.pt',
                             'ResNet34d_CBAM_best_fold_3.pt', 'ResNet34d_CBAM_best_fold_4.pt']
        self.model_name = 'ResNet34d_CBAM'
        self.device = torch.device("cpu")
        self.tta = True
        self.multi_folds = True
        self.img_size = 448
        self.multi_models = False


    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        dir_path = '/home/ai01/disk_7T/renlu/w/sjf/uwf/checkpoints'
        if not self.multi_folds:
            if self.model_name == '50d':
                self.model1 = ResNet50d(num_classes=2)
                checkpoint_path = os.path.join(dir_path, self.checkpoint1)
            elif self.model_name == '34d':
                self.model1 = ResNet34d(num_classes=2)
                checkpoint_path = os.path.join(dir_path, self.checkpoint2)
            elif self.model_name == 'GhostNet':
                self.model1 = GhostNet(num_classes=2)
                checkpoint_path = os.path.join(dir_path, self.checkpoint3)
            elif self.model_name == 'MobileNet':
                self.model1 = MobileNet(num_classes=2)
                checkpoint_path = os.path.join(dir_path, self.checkpoint4)
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
                elif self.model_name == 'GhostNet':
                    tmp_mode = GhostNet(num_classes=2)
                    checkpoints = self.checkpoints3
                elif self.model_name == 'MobileNet':
                    tmp_mode = MobileNet(num_classes=2)
                    checkpoints = self.checkpoints4
                elif self.model_name == 'vanb1':
                    tmp_mode = van_b1()
                    checkpoints = self.checkpoints5
                elif self.model_name == 'vanb2':
                    tmp_mode = van_b2()
                    checkpoints = self.checkpoints6
                elif self.model_name == 'ResNet34d_CBAM':
                    tmp_mode = ResNet34d_CBAM(num_classes=2)
                    checkpoints = self.checkpoints7
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
                for model_name in ['ResNet50d', 'ResNet34d']:
                    if model_name == 'ResNet50d':
                        tmp_mode = ResNet50d(num_classes=2)
                        checkpoints = self.checkpoints1
                    elif model_name == 'ResNet34d':
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
            ])
        ]


    def test(self):
        image = cv2.imread(os.path.join('/home/ai01/disk_7T/renlu/w/sjf/UWF4DR/ReferableDRandDME/1. Images/1. Training',
                                        'uwf4dr_rdr_dme_train_1.jpg'), 1)
        print(self.predict(image))


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)


class ResNet34d(nn.Module):
    def __init__(self, model_name='resnet34d', pretrained=False, num_classes=2, dropout=0.2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.get_classifier().in_features #512
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet34d_CBAM(nn.Module):
    def __init__(self, model_name='resnet34d', pretrained=False, num_classes=2, dropout=0.2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.get_classifier().in_features #512
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.ca = ChannelAttention(512)
        self.sa = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):
        out = self.feature(x)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class ResNet50d(nn.Module):
    def __init__(self, model_name='resnet50d', pretrained=False, num_classes=2, dropout=0.0):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        # self.model.fc = nn.Sequential(
        #     nn.Linear(num_features, 1024, bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(1024, num_classes, bias=True)
        # )

    def forward(self, x):
        x = self.model(x)
        return x


class GhostNet(nn.Module):
    def __init__(self, model_name='ghostnet_130', pretrained=False, num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, model_name='mobilenetv3_small_100', pretrained=False, num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        logger.info('model name is: ' + model_name)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNext(nn.Module):
    def __init__(self, model_name='resnext50d_32x4d', pretrained=False, num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class SeResNext(nn.Module):
    def __init__(self, model_name='seresnext50_32x4d', pretrained=False, num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


class UWFDataset(Dataset):
    def __init__(self, data, labels, name, transform=None, mode='train'):
        self.data = data
        self.labels = labels
        self.name = name
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.get_image(idx)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        # image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        label = self.labels[idx]
        return image[0], label

    def get_image(self, idx):
        if self.name in ['DME', 'RDR']:
            data_root = '/home/ai01/disk_7T/renlu/w/sjf/UWF4DR/ReferableDRandDME/1. Images/1. Training'
            image_name = self.data[idx]
            image = cv2.imread(os.path.join(data_root, image_name), 1)
            return image
        elif self.name == 'IQA':
            data_root1 = '/home/ai01/disk_7T/renlu/w/sjf/UWF4DR/ImageQualityAssessment/Images/Training'
            data_root2 = '/home/ai01/disk_7T/renlu/w/sjf/UWF4DR/ReferableDRandDME/1. Images/1. Training'
            image_name = self.data[idx]
            if image_name.startswith('uwf4dr_rdr'):
                image = cv2.imread(os.path.join(data_root2, image_name), 1)
            else:
                image = cv2.imread(os.path.join(data_root1, image_name), 1)
            return image

    def get_img_name(self, idx):
        return self.data[idx]


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        target = target.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = nn.functional.binary_cross_entropy_with_logits(x, target)
        return loss


def get_transforms(img_size=512, model_name='ResNet34d',mode='train'):
    if mode == 'train':
        if model_name in ['ResNet34d']:
            return Compose([
                Resize(img_size, img_size),
                # RandomResizedCrop(img_size, img_size, scale=(0.9, 1.0)),
                HorizontalFlip(p=0.5),
                # ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
                # RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.2),
                # A.CLAHE(clip_limit=(1, 4), p=0.5),
                MixUp(alpha=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return Compose([
                Resize(img_size, img_size),
                HorizontalFlip(p=0.5),
                MixUp(alpha=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    elif mode == 'val' or mode == 'test':
        return Compose([
            Resize(img_size, img_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def get_dataset(name, mode, model_name):
    if not name in ['DME', 'RDR', 'IQA']:
        print('we can not find dataset named ' + name + ' . you need to pass DME/RDR/IQA.')
        exit()
    csv_paths = {
        'DME' : '/home/ai01/disk_7T/renlu/w/sjf/UWF4DR/ReferableDRandDME/2. Groundtruths/1. Training.csv',
        'RDR' : '/home/ai01/disk_7T/renlu/w/sjf/UWF4DR/ReferableDRandDME/2. Groundtruths/1. Training.csv',
        'IQA' : '/home/ai01/disk_7T/renlu/w/sjf/UWF4DR/ImageQualityAssessment/Groundtruths/1. Training.csv'
    }
    df = pd.read_csv(csv_paths[name])
    if name == 'DME':
        data = df['image'].values
        label = df['diabetic macular edema'].values
        idx = np.where(pd.notna(label))[0]
        data = data[idx]
        label = label[idx]
    elif name == 'RDR':
        data = df['image'].values
        label = df['referable diabetic retinopathy'].values
    elif name == 'IQA':
        data = df['image'].values
        label = df['image quality level'].values

    return UWFDataset(data, label, name, get_transforms(args.image_size, model_name, mode), mode)


def train_one_model(my_args, model_name):
    dataset = get_dataset(name=my_args.task_name, mode=my_args.mode, model_name=model_name)
    kf = StratifiedKFold(n_splits=my_args.k, shuffle=True, random_state=42)
    train_loaders = []
    test_loaders = []
    # 生成K折的dataloaders
    for train_index, test_index in kf.split(dataset.data, dataset.labels):
        train_sampler = SubsetRandomSampler(train_index)
        test_sampler = SubsetRandomSampler(test_index)

        train_loader = DataLoader(dataset, batch_size=my_args.train_batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=my_args.test_batch_size, sampler=test_sampler)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    if args.multi_folds == 1:
        auc_list = []
        for i in range(args.k):
            auc = train_one_fold(train_loaders[i], test_loaders[i], i+1, my_args, model_name)
            auc_list.append(auc)
        logger.info('model: ' + model_name + '-----ave-AUC: ' + str(np.mean(auc_list)))
    else:
        train_one_fold(train_loaders[0], test_loaders[0], 0 + 1, my_args, model_name)


def train_one_fold(train_dataloader, test_dataloader, fold_cnt, my_args, model_name):
    pre_trained = False
    if my_args.pre_trained == 1:
        pre_trained = True
    if model_name == 'ResNet34d':
        model = ResNet34d(pretrained=pre_trained, num_classes=my_args.num_classes)
    elif model_name == 'ResNet50d':
        model = ResNet50d(pretrained=pre_trained, num_classes=my_args.num_classes)
    elif model_name == 'GhostNet':
        model = GhostNet(pretrained=pre_trained, num_classes=my_args.num_classes)
    elif model_name == 'MobileNet':
        model = MobileNet(pretrained=pre_trained, num_classes=my_args.num_classes)
    elif model_name == 'ResNext':
        model = ResNext(pretrained=pre_trained, num_classes=my_args.num_classes)
    elif model_name == 'SeResNext':
        model = SeResNext(pretrained=pre_trained, num_classes=my_args.num_classes)
    elif model_name == 'vanb1':
        model = van_b1(pretrained=pre_trained)
    elif model_name == 'vanb2':
        model = van_b2(pretrained=pre_trained)
    elif model_name == 'ResNet34d_CBAM':
        model = ResNet34d_CBAM(pretrained=pre_trained, num_classes=my_args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.label_smoothing == 1:
        criterion = LabelSmoothing()
    else:
        criterion = nn.BCEWithLogitsLoss()
    learning_rate = my_args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoches = my_args.epochs

    # train
    val_auc_list = []
    best_epoch = 0
    best_AUC = 0
    out_dir = my_args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info('TRAIN-----model: '+ model_name +'-----fold: ' + str(fold_cnt) + '-----')
    for epoch in range(0, epoches):
        logger.info('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        y_true = []
        y_scores = []
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)

            images, labels = images.to(device), labels.to(device)
            images = images.float()

            optimizer.zero_grad()
            outputs = model(images)

            labels_one_hot = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=my_args.num_classes).float()
            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()

            y_true.extend(labels.detach().cpu().numpy())
            prob = torch.softmax(outputs, dim=1).squeeze(0)
            if len(prob.shape) > 1:
                class_1_prob = prob[:, 1]
            else:
                class_1_prob = torch.Tensor([prob[1]])
            y_scores.extend(class_1_prob.detach().cpu().numpy())

            logger.info('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))

        logger.info('TRAIN-----model: ' + model_name + '-----fold: ' + str(fold_cnt) + '-----AUC: '
                    + str(roc_auc_score(y_true, y_scores)))


        logger.info('\nVAL-----model: ' + model_name + '-----fold: ' + str(fold_cnt) + '-----')
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            y_true = []
            y_scores = []
            for batch_idx, (images, labels) in enumerate(test_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                images = images.float()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum()

                y_true.extend(labels.detach().cpu().numpy())
                prob = torch.softmax(outputs, dim=1).squeeze(0)
                if len(prob.shape) > 1:
                    class_1_prob = prob[:, 1]
                else:
                    class_1_prob = torch.Tensor([prob[1]])
                y_scores.extend(class_1_prob.detach().cpu().numpy())

            logger.info('Val\'s acc is: %.3f%%' % (100 * correct / total))
            logger.info('Val\'s auc is: %.3f%%' % (100 * roc_auc_score(y_true, y_scores)))
            auc_val = roc_auc_score(y_true, y_scores)
            val_auc_list.append(auc_val)

        torch.save(model.state_dict(), out_dir + model_name + "_last_fold_"+ str(fold_cnt) +".pt")
        if auc_val == max(val_auc_list):
            torch.save(model.state_dict(), out_dir + model_name + "_best_fold_"+ str(fold_cnt) +".pt")
            logger.info("save epoch {} model".format(epoch))
            best_epoch = epoch
            best_AUC = auc_val

    logger.info('VAL-----model: ' + model_name + '-----fold: '
                + str(fold_cnt) + '\n-----BEST-EPOCH: ' + str(best_epoch) + '\n-----BEST-AUC: ' + str(best_AUC))
    return best_AUC


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--out_dir", default="checkpoints/", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")


    parser.add_argument("--model_name", default=None, type=str, required=False,
                        help="What model to use in training!")
    parser.add_argument("--image_size", default=512, type=int, required=False,
                        help="The image size using!")
    parser.add_argument("--num_classes", default=2, type=int, required=False,
                        help="class numbers")
    parser.add_argument("--task_name", default=None, type=str, required=False,
                        help="DME IQA RDR")


    parser.add_argument("--lr", default=1e-3, type=float, required=False,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--epochs", default=1, type=int, required=False,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--train_batch_size', default=4, type=int, required=False,
                        help='Train batch size')
    parser.add_argument('--test_batch_size', default=4, type=int, required=False,
                        help='Val batch size')

    parser.add_argument('--k', default=5, type=int, required=False,
                        help='fold number')
    parser.add_argument('--multi_folds', default=0, type=int, required=False,
                        help='k fold or not')
    parser.add_argument('--multi_models', default=0, type=int, required=False,
                        help='k fold or not')
    parser.add_argument('--label_smoothing', default=0, type=int, required=False,
                        help='label smoothing or not')
    parser.add_argument('--pre_trained', default=0, type=int, required=False,
                        help='label smoothing or not')



    parser.add_argument('--num_workers', default=4, type=int, required=False, help='DataLoader num_workers')
    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='accumulate the grad to update')
    parser.add_argument("--seed", type=int, default=42, required=False,
                        help="random seed for initialization")
    parser.add_argument("--mode", default='train', type=str, required=False,
                        help="train or test")

    return parser.parse_args()


def get_logger(logs_dir):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_file_path = os.path.join(logs_dir, 'train.log')
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(log_format)
    handler2 = FileHandler(filename=log_file_path)
    handler2.setFormatter(log_format)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def main(my_args):
    seed_everything(my_args.seed)
    if my_args.multi_models == 1:
        for model_name in ['ResNext', 'SeResNext']:
            train_one_model(my_args, model_name)
    else:
        train_one_model(my_args, my_args.model_name)


def test():
    my_model = Model()
    my_model.load(1)
    my_model.test()


def val(my_args):
    seed_everything(my_args.seed)
    dataset = get_dataset(name=my_args.task_name, mode='val', model_name=my_args.model_name)
    val_loader = DataLoader(dataset, batch_size=1)
    checkpoint1 = "ResNet50d_best_fold_1.pt"
    checkpoint2 = "ResNet34d_best_fold_1.pt"
    dir_path = '/home/ai01/disk_7T/renlu/w/sjf/uwf/checkpoints'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if my_args.model_name == 'ResNet34d':
        model = ResNet34d(num_classes=my_args.num_classes)
        checkpoint = checkpoint2
    elif my_args.model_name == 'ResNet50d':
        model = ResNet50d(num_classes=my_args.num_classes)
        checkpoint = checkpoint1
    checkpoint_path = os.path.join(dir_path, checkpoint)
    state_dict = torch.load(checkpoint_path)
    if any(key.startswith("module.") for key in state_dict.keys()):
        new_state_dict = {key[7:]: value for key, value in state_dict.items()}
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    bad_cases = []

    for idx, (images,labels) in enumerate(val_loader):
        img_name = dataset.get_img_name(idx)
        model.eval()
        images = images.to(device)
        images = images.float()
        outputs = model(images)
        prob = torch.softmax(outputs, dim=1).squeeze(0)
        if (prob[1] >0.5 and labels[0]==0) or (prob[1] < 0.5 and labels[0] == 1):
            bad_cases.append(img_name + ' ' + str(labels[0]))
    for i in bad_cases:
        print(i)
    logger.info(str(bad_cases))


def get_devices():
    num_gpu = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpu}")
    # 获取每个GPU的名称和性能信息
    for i in range(num_gpu):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # Convert bytes to GB
        print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} GB")


def show_data_aug():
    path = '/home/ai01/disk_7T/renlu/w/sjf/UWF4DR/ImageQualityAssessment/Images/Training'
    f1 = os.path.join(path, 'uwf4dr_iqa_train_1.jpg')
    f2 = os.path.join(path, 'uwf4dr_iqa_train_8.jpg')
    i1 = cv2.imread(f1)
    i2 = cv2.imread(f2)
    mixup = Compose([
        MixUp(p=1.0)
    ])
    mixed = mixup(image=i1, image0=i2)
    # 获取混合后的图像
    mixed_image = mixed['image']
    # 显示混合后的图像
    cv2.imshow('Mixed Image', mixed_image)

if __name__ == '__main__':
    # get_devices()
    # pretrain
        # 加入环境变量
        # export HF_ENDPOINT=https://hf-mirror.com
        # 代码开头加入
        # import os
        # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    torch.cuda.set_device(0)
    args = get_args()
    args.model_name = 'ResNet34d_CBAM'
    args.image_size = 512
    args.num_classes = 2
    args.task_name = 'IQA'
    args.lr = 6e-5
    args.epochs = 80
    args.train_batch_size = 8
    args.test_batch_size = 4
    args.k = 4
    args.seed = 42
    args.multi_folds = 1 # 1:训练K折 其他值:训练一折
    args.multi_models = 0  # 1:训练多个 其他值:训练一个
    args.label_smoothing = 1
    args.pre_trained = 0
    logger = get_logger(args.out_dir)
    # main(args)
    test()