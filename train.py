import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as F
import numpy as np
import cv2
import os
import csv


class MicroPlasticDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.img_paths = []
        self.boxes = []
        self.labels = []

        # Read annotations
        with open(data_dir, 'r') as r:
            reader = csv.reader(r)
            # Skip header
            next(reader)
            for row in reader:
                self.img_paths.append(row[0])
                # x1, y1, x2, y2
                self.boxes.append([int(row[-4]), int(row[-3]), int(row[-2]), int(row[-1])])
                self.labels.append(1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        box = self.boxes[idx]
        box = torch.tensor(box, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        return img, {'boxes': box, 'labels': label}


if __name__ == '__main__':
    training_dataset = MicroPlasticDataset('data/train.csv')
    validation_dataset = MicroPlasticDataset('data/val.csv')
    train_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=False)

    for img, target in train_dataloader:
        print(img.shape)
        print(target)
        break
