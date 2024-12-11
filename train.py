import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as F
import numpy as np
import cv2
import os
import csv
from alive_progress import alive_bar

class MicroPlasticDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.img_paths = []
        self.boxes = []
        self.labels = []
        annotations_path = f'{data_dir}/annotations.csv'

        # Read annotations
        with open(annotations_path, 'r') as r:
            reader = csv.reader(r)
            # Skip header
            next(reader)
            for row in reader:
                self.img_paths.append(f'{data_dir}/{row[0]}')
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


def train_one_epoch(model, train_dataloader, optimizer, device):
    training_losses = []
    model.train()
    print("Training...")
    with alive_bar(len(train_dataloader)) as bar:
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            images = list(image.to(device) for image in images)

            # targets = {'boxes'=tensor, 'labels'=tensor}
            targets = [{box: label.to(device) for box, label in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            training_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            bar()
    return training_losses


def validate(model, val_dataloader, device):
    model.to(device)
    validation_loss_list = []
    with alive_bar(len(val_dataloader)) as bar:
        model.eval()
        for images, targets in val_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            validation_loss_list.append(losses.item())
            bar()
    return validation_loss_list


if __name__ == '__main__':
    data_dir = '/media/alex/1TBSSD/SSD/Microplastic_Dataset/'
    training_dataset = MicroPlasticDataset(data_dir + 'train')
    validation_dataset = MicroPlasticDataset(data_dir + 'val')
    train_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=False)

    for img, target in train_dataloader:
        print(img.shape)
        print(target)
        break

    SEED = 42
    LR = 0.005
    LR_MOMENTUM = 0.9
    LR_DECAY_RATE = 0.0005

    LR_SCHED_STEP_SIZE = 0.1
    LR_SCHED_GAMMA = 0.1
    NUM_EPOCHS = 5

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Micro plastics and background
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    torch.manual_seed(SEED)
    
    for epoch in range(NUM_EPOCHS):
        print("----------Epoch {}----------".format(epoch + 1))
        training_loss = train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        lr_scheduler.step()
        print("Training loss: {:.2f}".format(np.mean(training_loss)))
        validation_loss = validate(model, val_dataloader, device)
        print("Validation loss: {:.2f}".format(np.mean(validation_loss))

    torch.save(model.state_dict(), 'faster_rcnn.pth')