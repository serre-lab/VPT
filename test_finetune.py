import os
import argparse
import numpy as np
import timm
import wandb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from src.perspective_data import ViewDataset
from src.utils import binary_accuracy, get_args_parser, get_transform_wo_crop
import json
import csv
from pathlib import Path

def evaluate_exp(model, data_loader, device, args):
    model.eval()
    preds_list = []
    preds_list_logits = []
    labels_list = []
    img_path_list = []
    
    for i, batch in enumerate(data_loader):
        imgs, labels, img_path = batch
        imgs = imgs.to(device)
        labels = labels.float().to(device)
        labels = torch.unsqueeze(labels, 1)
        preds = model(imgs)
        img_path_list += img_path
        preds_list.append(preds.detach())
        preds_list_logits.append(preds.detach())
        labels_list.append(labels)
    img_path_record = np.array(img_path_list).squeeze().T
    preds_record = torch.cat(preds_list).cpu().numpy().squeeze().T
    labels_record = torch.cat(labels_list).cpu().numpy().squeeze().T
    records = np.vstack([img_path_record, preds_record, labels_record]).T
    
    preds = torch.cat(preds_list_logits).squeeze().cpu()
    labels = torch.cat(labels_list).squeeze().cpu()
    epoch_acc = binary_accuracy(preds, labels)
    return epoch_acc, records

def run_evaluate(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    model = timm.create_model(args.model_name, pretrained=(not args.not_pretrained), num_classes=args.num_classes, drop_rate=args.dropout_rate)
    model.load_state_dict(torch.load(args.ckpt_path).state_dict())
    model = model.to(device)
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)
    dataset = ViewDataset(args.data_dir, transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    acc, records = evaluate_exp(model, data_loader, device, args)
    logs_dir = 'logs/finetune_view_exp/'
    os.makedirs(logs_dir, exist_ok=True)
    if os.path.exists(os.path.join(logs_dir, 'view_results.json')):
        with open(os.path.join(logs_dir, 'view_results.json'), 'r') as f:
            results = json.load(f)
    else:
        results = {}
    results[args.model_name] = acc
    print(args.model_name, acc)
    with open(os.path.join(logs_dir, 'view_results.json'), 'w') as f:
        results_json = json.dumps(results, indent=4)
        f.write(results_json)
    with open(os.path.join(logs_dir, f'{args.model_name}_view_ft.csv'), 'w') as f:
        header = ['path', 'preds', 'label']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records.tolist())
    
if __name__ == '__main__':
    args_parser = get_args_parser()
    #args_parser.add_argument('--ckpt_path', type=str, help="Model Checkpoint Path")
    args = args_parser.parse_args()
    run_evaluate(args)