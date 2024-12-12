import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import csv
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import optuna
import logging
import sys
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
preprocess = weights.transforms()

class MicroPlasticDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.images = []
        annotations_path = f'{data_dir}/annotations.csv'
        data = {}
        # Read annotations
        with open(annotations_path, 'r') as r:
            reader = csv.reader(r)
            # Skip header
            next(reader)
            for row in reader:
                img = data.get(f'{data_dir}/{row[0]}', {})
                img['boxes'] = img.get('boxes', [])
                img['labels'] = img.get('labels', [])
                img['boxes'].append([int(row[-4]), int(row[-3]), int(row[-2]), int(row[-1])])
                img['labels'].append(1)
                data[f'{data_dir}/{row[0]}'] = img

        for img_path, img_data in data.items():
            self.images.append((img_path, img_data['boxes'], img_data['labels']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.images[idx]
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img /= 255.0
        # img = torch.from_numpy(img).permute(2, 0, 1)
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        # img = preprocess(img)
        img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float32)

        img_transforms = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
        ])
        # img = img_transforms(img)

        # Boxes: Shape -> [N, 4] with N = Number of Boxes
        boxes = torch.tensor(boxes, dtype=torch.float32)
        #boxes = tv_tensors.BoundingBoxes(boxes, format='xyxy', canvas_size=(img_width, img_height), dtype=torch.float32)
        # Labels: Shape -> [N] with N = Number of Boxes
        label = torch.tensor(labels, dtype=torch.int64)
        #img, boxes = img_transforms(img, boxes)
        return img, {'boxes': boxes, 'labels': label}


def train_one_epoch(model, train_dataloader, optimizer, device):
    training_losses = []
    model.train()
    print("Training...")
    with alive_bar(len(train_dataloader), force_tty=True) as bar:
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            images = list(image.to(device) for image in images)

            # targets = {'boxes'=tensor, 'labels'=tensor}
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            training_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            bar()
    return training_losses

def collate_fn(batch):
  return tuple(zip(*batch))


def validate(model, val_dataloader, device, trial=None):
    model.to(device)
    validation_score_list = []
    validation_accuracy_list = []
    validation_precision_list = []
    validation_recall_list = []
    with alive_bar(len(val_dataloader), force_tty=True) as bar:
        model.eval()
        for images, labels in val_dataloader:
            images = list(image.to(device) for image in images)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
            results = model(images)
            scores = []
            for img_res in results:
                score = np.mean(img_res['scores'].detach().cpu().numpy())
                scores.append(score)
            validation_score_list.extend(scores)
            # print(f'Validation Score: {np.mean(scores)}')
            avg_accuracy = 0
            avg_precision = 0
            avg_recall = 0
            for i, img_res in enumerate(results):
                accuracy, precision, recall  = get_accuracy(img_res['boxes'].detach().cpu().numpy().tolist(),
                                                            img_res['labels'].detach().cpu().numpy().tolist(),
                                                            labels[i]['boxes'].detach().cpu().numpy().tolist())
                if accuracy > 1:
                    print(f'Error')
                    raise ValueError('Accuracy cannot be greater than 1: {}'.format(accuracy))
                avg_accuracy += accuracy
                avg_precision += precision
                avg_recall += recall
            avg_accuracy /= len(results)
            avg_precision /= len(results)
            avg_recall /= len(results)
            validation_accuracy_list.append(avg_accuracy)
            validation_precision_list.append(avg_precision)
            validation_recall_list.append(avg_recall)
            bar()
    return validation_score_list, validation_accuracy_list, validation_precision_list, validation_recall_list

def get_accuracy(predicted_boxes, predicted_labels, ground_truth_boxes):
    if len(predicted_boxes) == 0:
        return 0, 0, 0

    true_positives = 0
    # Match each ground truth box with a predicted box
    for gt_box in ground_truth_boxes:
        for idx, pred_box in enumerate(predicted_boxes):
            # check only boxes labeld as microplastic
            if predicted_labels[idx] == 1:
                iou = bb_intersection_over_union(gt_box, pred_box)
                if iou > 0.8:
                    true_positives += 1
                    break

    false_positives = len(predicted_boxes) - true_positives
    false_negatives = len(ground_truth_boxes) - true_positives
    accuracy = true_positives / (true_positives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return accuracy, precision, recall


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def objective(trial):
    data_dir = '/media/alex/1TBSSD/SSD/Microplastic_Dataset/'
    training_dataset = MicroPlasticDataset(data_dir + 'train')
    validation_dataset = MicroPlasticDataset(data_dir + 'val')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    SEED = 42


    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Micro plastics and background
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.009, log=True)
    momentum = trial.suggest_float('momentum', 0.8, 0.9)
    step_size = trial.suggest_float('step_size', 5, 10)
    weight_decay = trial.suggest_float('weight_decay', 0.0005, 0.0006)
    num_epochs = trial.suggest_int('num_epochs', 10, 35)

    print(f'Learning Rate: {learning_rate}, Step Size: {step_size}, Momentum: {0.9},'
          f' Weight Decay: {weight_decay}, Num Epochs: {num_epochs}')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(params, lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    torch.manual_seed(SEED)

    validation_accuracy_list = []
    training_accuracy_list = []
    validation_precision_list = []
    for epoch in range(num_epochs):
        print("----------Epoch {}----------".format(epoch + 1))
        training_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        lr_scheduler.step()
        print('Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
        print("Training loss: {:.2f}".format(np.mean(training_loss)))
        validation_loss, validation_accuracy, validation_precision, validation_recall = validate(model, val_dataloader, device)
        validation_accuracy_list.extend(validation_accuracy)
        training_accuracy_list.extend(training_loss)
        validation_precision_list.extend(validation_precision)
        print("Average Validation score: {:.2f}".format(np.mean(validation_loss)))
        print("Average Validation Accuracy: {:.2f}".format(np.mean(validation_accuracy)))
        print("Average Validation Precision: {:.2f}".format(np.mean(validation_precision)))
        print("Average Validation Recall: {:.2f}".format(np.mean(validation_recall)))
        trial.report(np.mean(validation_accuracy), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        draw_boxes(model, epoch, device, trial.number)
    return np.mean(validation_accuracy_list)

def draw_boxes(model, epoch, device, trial_num=0):
    model.eval()
    img = Image.open('/media/alex/1TBSSD/SSD/Microplastic_Dataset/train/a--100-_jpg.rf.77ca389cf7e6997bcae4d1bc8c86044a.jpg').convert('RGB')
    # img_transforms = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # img_tensor = img_transforms(img).to(device)
    img_tensor = preprocess(img).to(device)
    results = model([img_tensor])
    boxes = results[0]['boxes'].detach().cpu().numpy().tolist()
    if len(boxes) > 0:
        img = np.array(img)
        for box in boxes:
            x1, y1, x2, y2 = box
            img = cv2.rectangle(img, (int(x1), int(x2)), (int(x2), int(y2)), (255, 255, 0), 2)
        Path('test_images').mkdir(exist_ok=True)
        cv2.imwrite(f'test_images/trial_{trial_num}_epoch_{epoch}.jpg', img)

if __name__ == '__main__':
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///faster-rcnn-resnet-tuning.db"
    study = optuna.create_study(directions=['maximize'],
                                sampler=optuna.samplers.TPESampler(),
                                storage=storage_name,
                                load_if_exists=True,
                                study_name='faster-rcnn-resnet-tuning')
    study.set_metric_names(['accuracy'])
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    best_resnet_trial = study.best_trial

    print("Accuracy: ", best_resnet_trial.value)

    print("  Params: ")
    for key, value in best_resnet_trial.params.items():
        print("    {}: {}".format(key, value))

    # torch.save(model.state_dict(), 'faster_rcnn.pth')
    # plt.plot(training_accuracy_list, label='Training Loss')
    # plt.plot(validation_accuracy_list, label='Validation Accuracy')
    # plt.legend()
    # plt.show()
    # plt.savefig('accuracy.png')