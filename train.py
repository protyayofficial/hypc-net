import os
import argparse
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from PIL import ImageFile
import time
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
import warnings

import numpy as np
import random
import csv

from utils import initialize_model
from utils import SafeImageFolder
from utils import generate_meta_features
from utils import write_csv

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Transfer learning for HYPC Net')
parser.add_argument('--name', default='dis_vgg16', type=str, help='name of the run')
parser.add_argument('--seed', default=9, type=int, help='random seed')
parser.add_argument('--train', default='data/train.tsv', type=str, help='path to train image files/labels')
parser.add_argument('--dev', default='data/dev.tsv', type=str, help='path to dev image files/labels')
parser.add_argument('--test', default='data/test.tsv', type=str, help='path to test image files/labels')
parser.add_argument('--out-file', default='out/results.json', type=str, help='path to output file')
parser.add_argument('--sep', default='\t', type=str, help='column separator used in csv(default: "\t")')
parser.add_argument('--data-dir', default='./', type=str, help='root directory of images')
parser.add_argument('--best-state-path', default='models/best.pth', type=str, help='path to best state checkpoint')
parser.add_argument('--fig-dir', default='out/figures', type=str, help='directory path for output figures')
parser.add_argument('--checkpoint-dir', default='out/models', type=str, help='directory for output models/states')
parser.add_argument('--arch', default='vgg16', type=str,
                    help='model architecture [resnet18, resnet50, resnet101, alexnet, vgg, vgg16, squeezenet, densenet,'
                         'inception, efficientnet-b1, efficientnet-b7, convnext] (default: resnet18)')
parser.add_argument('--yoga-class', default=82, type=int, help='number of classes for evaluation (6, 20, 82)')
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--learning-rate', default=1e-5, type=float, help='initial learning rate (default: 1e-5)')
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay (default: 0)')
parser.add_argument('--num-epochs', default=50, type=int, help='number of epochs(default: 50)')
parser.add_argument('--use-rand-augment', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='use random augment or not')
parser.add_argument('--keep-frozen', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to keep feature layers frozen (i.e., only update classification layers weight)')
parser.add_argument('--rand-augment-n', default=2, type=int,
                    help='random augment parameter N or number of augmentations applied sequentially')
parser.add_argument('--rand-augment-m', default=9, type=int,
                    help='random augment parameter M or shared magnitude across all augmentation operations')


def set_seed(seed: int = 9):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, yoga_class, dataset_sizes):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Open the CSV file in write mode and write the header
    with open(f"{model}_{yoga_class}_training_metrics.csv", 'w', newline='') as output_file:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_metrics = {'epoch': epoch}

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                if inputs is None:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc'] = epoch_acc.item()

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'{model}_{yoga_class}_new_best_model.pth')

        # Write metrics to CSV after each epoch
        with open(f"{model}_{yoga_class}_new_metrics.csv", 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            dict_writer.writerow(epoch_metrics)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, device, class_names):
    meta_features, meta_labels = generate_meta_features(model, dataloaders, device)
    all_meta_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            meta_features = []

            outputs = model(inputs)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            meta_features.append(preds)

            meta_features = np.hstack(meta_features)
            all_meta_features.append(meta_features)
            all_labels.append(labels.cpu().numpy())

    meta_features = np.vstack(all_meta_features)
    meta_labels = np.hstack(all_labels)
   #print(meta_features.shape)
    
    cb_model = cb.CatBoostClassifier(verbose=0,random_state=9)
    cb_model.fit(meta_features, meta_labels)

    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=9)
    xgb_model.fit(meta_features, meta_labels)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=9)
    rf_model.fit(meta_features, meta_labels)
    
    metrics = {
        'accuracy': {},
        'precision': {},
        'recall': {},
        'f1_score': {},
        'classification_report': {}
    }
    backend_dict = {
        'CatBoost': cb_model,
        'XGBoost': xgb_model,
        'RandomForest': rf_model
    }

    for backend_name ,backend in backend_dict.list():
        
        final_preds = backend.predict(meta_features)
        #final_preds_probs = backend.predict_proba(meta_features)

        accuracy = accuracy_score(meta_labels, final_preds)
        precision = precision_score(meta_labels, final_preds, average='weighted')
        recall = recall_score(meta_labels, final_preds, average='weighted')
        f1 = f1_score(meta_labels, final_preds, average='weighted')
        report = classification_report(meta_labels, final_preds, target_names=class_names)

        print(backend_name)
        print(f'Accuracy: {accuracy:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'F1 Score: {f1:.3f}', flush=True)

        metrics['accuracy'][backend_name] = accuracy
        metrics['precision'][backend_name] = precision
        metrics['recall'][backend_name] = recall
        metrics['f1_score'][backend_name] = f1
        metrics['classification_report'][backend_name] = report

    return metrics, meta_labels, final_preds

    
def main():

    use_gpu = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(use_gpu)
    print("Using device: {}".format(use_gpu), flush=True)
    args = parser.parse_args()
    print(args, flush=True)

    set_seed(args.seed)

    model_name = args.arch
    lr = args.learning_rate
    num_epochs = args.num_epochs

    model = initialize_model(model_name, yoga_class=args.yoga_class, keep_frozen=args.keep_frozen, use_pretrained=True)
    model = model.to(device)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),    
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...", flush=True)

    image_datasets = {
    'train': SafeImageFolder("f'{args.data_dir}/yoga_train/class_{args.yoga_class}'", data_transforms['train']),
    'test': SafeImageFolder("f'{args.data_dir}/yoga_test/class_{args.yoga_class}'", data_transforms['test']),
}

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=16, shuffle=False, num_workers=4),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device=device,
                                      num_epochs=num_epochs, yoga_class=args.yoga_class, dataset_sizes=dataset_sizes)


    if args.test:
        results = test_model(model, dataloaders, device=device, class_names=class_names)
        write_csv(results, args.out_dir, yoga_class=args.yoga_class)
if __name__ == "__main__":
    main() 