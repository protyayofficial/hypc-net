import os
import timm
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import csv

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                index = (index + 1) % len(self.samples)


def initialize_model(model_name, yoga_class, use_pretrained=True):
   
    model = None
    # model input image size
    input_image_size = 0
    # image is resized to this size before cropping as input_image_size
    resize_image_size = 0
    
    if model_name == "resnet50":
        model = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, yoga_class)
    
    elif model_name == "vgg16":
        model = models.vgg16_bn(pretrained=use_pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, yoga_class)
    
    elif model_name == "densenet":
        model = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, yoga_class)
    
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3(pretrained=use_pretrained)
        num_ftrs = model.last_channel
        model.fc = nn.Linear(num_ftrs, yoga_class)

    elif model_name == "efficientnet-b1":
        model = EfficientNet.from_pretrained('efficientnet-b1')
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, yoga_class)

    elif model_name == "efficientnet-b7":
        model = EfficientNet.from_pretrained('efficientnet-b7')
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, yoga_class)

    elif model_name == "swin_tiny":
        model = models.swin_t(pretrained = use_pretrained) 
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, yoga_class)

    elif model_name == "convnext":
        model = timm.create_model('convnext_base', pretrained=use_pretrained)
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, yoga_class)
    else:
        print("Invalid model name, exiting...", flush=True)
        exit()

    return model, resize_image_size, input_image_size

def generate_meta_features(model, dataloaders, device):
    
    model.eval()
    all_meta_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            meta_features = []

            outputs = model(inputs)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            meta_features.append(preds)

            meta_features = np.hstack(meta_features)
            all_meta_features.append(meta_features)
            all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_meta_features), np.hstack(all_labels)

def write_csv(metrics, out_dir, yoga_class):
    csv_file = f'test_{yoga_class}.csv'
    csv_file_path = os.path.join(out_dir, csv_file)

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

        for i in range(len(metrics['model'])):
            writer.writerow([
                metrics['model'][i],
                metrics['accuracy'][i],
                metrics['precision'][i],
                metrics['recall'][i],
                metrics['f1_score'][i]
            ])

    print(f"Metrics have been saved to '{csv_file}'")