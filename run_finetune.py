import os
import argparse
import numpy as np
import timm
import wandb
from tqdm import tqdm

import torch
#from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


from src.perspective_data import PerspectiveDatasetImageFolder, PerspectiveDataset
from src.utils import binary_accuracy, accuracy, CosineAnnealingWithWarmup, get_args_parser, get_transform_wo_crop
import json
import csv
from pathlib import Path

def train(model, train_loader, test_loader, val_loader, human_loader, criterion, optimizer, lr_scheduler, device, args):
    best_acc_test = 0
    best_acc_val = 0
    best_acc_train = 0
    best_acc_human = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_acc = []
        epoch_loss = []
        for i, batch in enumerate(train_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            labels = torch.unsqueeze(labels, 1)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            #acc = accuracy(preds,labels)[0]
            acc = binary_accuracy(preds, labels)
            epoch_acc.append(acc)
            epoch_loss.append(loss.item())
        with torch.no_grad():
            test_acc, test_loss, record = evaluate(model, test_loader, criterion, device, True, args)
            val_acc, val_loss, _ = evaluate(model, val_loader, criterion, device, False, args)
            human_acc, human_loss, human_record = evaluate(model, human_loader, criterion, device, True, args)
            train_acc = sum(epoch_acc)/float(len(epoch_acc))
            if val_acc > best_acc_val:
                best_acc_val = val_acc
                best_acc_test = test_acc
                best_acc_human = human_acc
                best_acc_train = train_acc
                if args.task == 'depth':
                    filename = "preds_depth"
                else:
                    filename = "preds"
                if not args.not_pretrained:
                    file_names = [f'{args.model_name}_{filename}_ft.csv', f'{args.model_name}_{filename}_human_ft.csv']
                else:
                    file_names = [f'{args.model_name}_{filename}_sc.csv', f'{args.model_name}_{filename}_human_sc.csv']
                with open(f'./logs/fine_tune_preds/{file_names[0]}', 'w') as f:
                    header = ['path', 'pred', 'label']
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(record.tolist())
                with open(f'./logs/fine_tune_preds/{file_names[1]}', 'w') as f:
                    header = ['path', 'pred', 'label']
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(human_record.tolist())
                torch.save(model, f'./logs/fine_tune_ckpts/{args.model_name}_{args.task}.ckpt')
                
        if args.wandb:
            wandb.log({'train_acc':train_acc, 'train_loss':sum(epoch_loss)/float(len(epoch_loss)),
                    'val_acc':val_acc, 'val_loss':val_loss, 'human_acc':human_acc, 'human_loss':human_loss,
                    'test_acc':test_acc, 'test_loss':test_loss, "lr": lr_scheduler.get_last_lr()[0]})
    return best_acc_train, best_acc_test, best_acc_val, best_acc_human
    
def evaluate(model, data_loader, criterion, device, return_record, args):
    model.eval()
    epoch_loss = []
    preds_list = []
    preds_list_logits = []
    labels_list = []
    img_path_list = []
    
    for i, batch in enumerate(data_loader):
        if return_record:
            imgs, labels, img_path = batch
        else:
            imgs, labels = batch
            img_path = None
            
        imgs = imgs.to(device)
        labels = labels.float().to(device)
        labels = torch.unsqueeze(labels, 1)
        preds = model(imgs)
        loss = criterion(preds, labels)
        epoch_loss.append(loss.item())
        
        if return_record:
            img_path_list+=img_path
            #preds_list.append(torch.sigmoid(preds))
            preds_list.append(preds)
        preds_list_logits.append(preds)
        labels_list.append(labels)
    
    if return_record:
        img_path_record = np.array(img_path_list).squeeze().T
        preds_record = torch.cat(preds_list).cpu().numpy().squeeze().T
        labels_record = torch.cat(labels_list).cpu().numpy().squeeze().T
        records = np.vstack([img_path_record, preds_record, labels_record]).T
    else:
        records = None
    preds = torch.cat(preds_list_logits).squeeze().cpu()
    labels = torch.cat(labels_list).squeeze().cpu()
    epoch_acc = binary_accuracy(preds, labels)
    return epoch_acc, sum(epoch_loss)/float(len(epoch_loss)), records
            
def run(args):
    #torch.set_float32_matmul_precision('high')
    if args.wandb:
        wandb.init(project='gs-perception-finetune', 
                            config={                   
                                "learning_rate": args.learning_rate,
                                "dropout_rate": args.dropout_rate,
                                "weight_decay": args.weight_decay,
                                "architecture": args.model_name,
                                "epochs": args.epochs,
                                } )
    device = torch.device(f'cuda:{args.gpu_id}')
    
    model = timm.create_model(args.model_name, pretrained=(not args.not_pretrained), num_classes=args.num_classes, drop_rate=args.dropout_rate)
    
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)
    # transforms_train = timm.data.create_transform(**data_config, is_training=True)
    # transforms = timm.data.create_transform(**data_config, is_training=False)
    if not args.flip:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform)
    else:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_flip'), transform=transform)
    test_dataset = PerspectiveDataset(Path(args.data_dir).parent, transform, split='test', task=args.task, return_path=True)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform)
    human_dataset = PerspectiveDataset(Path(args.data_dir).parent, transform, split='human', task=args.task, return_path=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    human_loader = DataLoader(human_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    #model = torch.compile(model)
    model = model.to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    steps_per_epoch = len(train_loader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                weight_decay=args.weight_decay, amsgrad=False)
    lr_scheduler = CosineAnnealingWithWarmup(optimizer = optimizer, min_lrs = [args.min_lr], first_cycle_steps = args.epochs*steps_per_epoch, warmup_steps = args.warmup*steps_per_epoch, gamma = 0.9)
    #lr_scheduler = None
    best_acc_train, best_acc_test, best_acc_val, best_acc_human = train(model, train_loader, test_loader, val_loader, human_loader, criterion, optimizer, lr_scheduler, device, args)
    print("Best acc train", best_acc_train)
    print("Best acc test", best_acc_test)
    print("Best acc val", best_acc_val)
    print("Best acc human", best_acc_human)
    if args.task == 'depth':
        filename = 'depth_results_ft.json'
    else:
        filename = 'perspective_results_ft.json'
    with open(f'logs/{filename}', 'r') as f:
        results = json.load(f)
    results[args.model_name] = [best_acc_train, best_acc_test, best_acc_val, best_acc_human]
    with open(f'logs/{filename}', 'w') as f:
        results_json = json.dumps(results, indent=4)
        f.write(results_json)
        
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run(args)
        
                                
                                