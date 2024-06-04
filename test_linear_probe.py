import os
import argparse
import numpy as np
import timm
import wandb
from tqdm import tqdm
import cv2
import torch
import pandas as pd
from pathlib import Path
#from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import Compose, ToTensor
from src.perspective_data import PerspectiveDataset, FeaturesDataset, ViewDataset
from src.models import LinearModel
from src.utils import binary_accuracy, CosineAnnealingWithWarmup, get_args_parser, get_transform_wo_crop
import json
import csv


def extract_features(model, data_loader, device, return_path = False):
    model.eval()
    features = []
    labels_list = []
    img_path_list = []
    for data in tqdm(data_loader):
        if return_path:
            images, labels, img_paths = data
            img_path_list+=(img_paths)
        else:
            images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(images)
            features.append(preds.cpu())
            labels_list.append(labels.cpu())
    features = torch.cat(features)
    labels = torch.cat(labels_list).squeeze()
    if not return_path:
        return features, labels
    else:
        return features, labels, img_path_list

def train_linear_probe(model, train_loader, val_loader, exp_loader, criterion, optimizer, lr_scheduler, device, args):
    best_acc_val = 0
    best_acc_train = 0
    best_acc_exp = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_acc = []
        epoch_loss = []
        for i, batch in enumerate(train_loader):
            features, labels = batch
            features = features.to(device)
            labels = labels.float().to(device)
            labels = torch.unsqueeze(labels, 1)
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            acc = binary_accuracy(preds, labels)
            epoch_acc.append(acc)
            epoch_loss.append(loss.item())
        with torch.no_grad():
            val_acc, val_loss = evaluate_linear_probe(model, val_loader, criterion, device, False, args)
            exp_acc, exp_loss, exp_record = evaluate_linear_probe(model, exp_loader, criterion, device, True, args)
            train_acc = sum(epoch_acc)/float(len(epoch_acc))
            if val_acc > best_acc_val:
                best_acc_val = val_acc
                best_acc_train = train_acc
                best_acc_exp = exp_acc
                file_name = "preds"
                file_name = f'{args.model_name}_{file_name}_view_lp.csv'
                with open(f'./logs/linear_probe_view_exp/{file_name}', 'w') as f:
                    header = ['path', 'pred', 'label']
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(exp_record.tolist())
        if args.wandb:
            wandb.log({'train_acc':train_acc, 'train_loss':sum(epoch_loss)/float(len(epoch_loss)),
                    'val_acc':val_acc, 'val_loss':val_loss, 'exp_acc':exp_acc, 'exp_loss':exp_loss})
    return best_acc_train, best_acc_val, best_acc_exp
    
def evaluate_linear_probe(model, data_loader, criterion, device, return_record, args):
    model.eval()
    epoch_loss = []
    preds_list = []
    labels_list = []
    preds_list_logits = []
    img_path_list = []
    for i, batch in enumerate(data_loader):
        if return_record:
            features, labels, img_path = batch
        else:
            features, labels = batch
            img_path =  None
        features = features.to(device)
        labels = labels.float().to(device)
        labels = torch.unsqueeze(labels, 1)
        preds = model(features)
        loss = criterion(preds, labels)
        preds_list_logits.append(preds)
        labels_list.append(labels)
        epoch_loss.append(loss.item())
        if return_record:
            img_path_list += img_path
            #preds_list.append(torch.sigmoid(preds))
            preds_list.append(preds)

    preds = torch.cat(preds_list_logits).squeeze().cpu()
    labels = torch.cat(labels_list).squeeze().cpu()
    epoch_acc = binary_accuracy(preds, labels)
    if return_record:
        img_path_record = np.array(img_path_list).squeeze().T
        preds_record = torch.cat(preds_list).cpu().numpy().squeeze().T
        labels_record = torch.cat(labels_list).cpu().numpy().squeeze().T
        records = np.vstack([img_path_record, preds_record, labels_record]).T
        return epoch_acc, sum(epoch_loss)/float(len(epoch_loss)), records
    return epoch_acc, sum(epoch_loss)/float(len(epoch_loss))

def run_extract_features(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
        
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)
    if not args.flip:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform)
    else:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_flip'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform)
    exp_dataset = ViewDataset(args.exp_dir, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True)
    exp_loader = DataLoader(exp_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True)
    model = model.to(device)
    
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    exp_features, exp_labels, exp_img_paths = extract_features(model, exp_loader, device, return_path=True)
    

    return train_features, train_labels, val_features, val_labels, exp_features, exp_labels, exp_img_paths
    
def run_linear_probe(args):
    if args.wandb:
        wandb_run = wandb.init(project='gs-perception-linear-probe', 
                            config={                   
                                "learning_rate": args.learning_rate,
                                "architecture": args.model_name,
                                "epochs": args.epochs,
                                } )
    device = torch.device(f'cuda:{args.gpu_id}')
    train_features, train_labels, \
    val_features, val_labels, \
    exp_features, exp_labels, exp_img_paths = run_extract_features(args)
        
    train_feat_dataset = FeaturesDataset(train_features, train_labels)
    val_feat_dataset = FeaturesDataset(val_features, val_labels)
    exp_feat_dataset = FeaturesDataset(exp_features, exp_labels, exp_img_paths)
    
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
    val_feat_loader = DataLoader(val_feat_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    exp_feat_loader = DataLoader(exp_feat_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    criterion = torch.nn.BCEWithLogitsLoss()
    linear_model = LinearModel(train_features.shape[-1], args.num_classes, args.dropout_rate)
    optimizer = torch.optim.AdamW(linear_model.parameters(), lr=args.learning_rate,
                                weight_decay=args.weight_decay, amsgrad=False)
    lr_scheduler = None
    linear_model = linear_model.to(device)
    best_acc_train, best_acc_val, best_acc_exp = train_linear_probe(linear_model, train_feat_loader, 
                                                                    val_feat_loader, exp_feat_loader, 
                                                                    criterion, optimizer, lr_scheduler, device, args)
    print("Best acc validation", best_acc_val)
    print("Best acc train", best_acc_train)
    print("Best acc human", best_acc_exp)
    logs_dir = './logs/linear_probe_view_exp/'
    log_file = f'view_results.json'
    if os.path.exists(os.path.join(logs_dir, log_file)):
        with open(os.path.join(logs_dir, log_file), 'r') as f:
            results = json.load(f)
    else:
        results = {}
    results[args.model_name] = [best_acc_train, best_acc_val, best_acc_exp]
    with open(os.path.join(logs_dir, log_file), 'w') as f:
        results_json = json.dumps(results, indent=4)
        f.write(results_json)
    if args.wandb:
        wandb_run.finish()
        
if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    run_linear_probe(args)
        
                                
                                