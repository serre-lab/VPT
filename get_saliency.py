import os
import argparse
import numpy as np
import timm
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from submodule.pytorch_smoothgrad.lib.gradients import SmoothGrad
from submodule.pytorch_smoothgrad.lib.image_utils import save_diff_map, save_as_heatmap, save_as_overlay
from src.utils import binary_accuracy, CosineAnnealingWithWarmup, get_args_parser, get_transform_wo_crop
from PIL import Image
from src.models import LinearModel, LinearProbeModel
from run_linear_probe import extract_features, evaluate_linear_probe
from src.perspective_data import PerspectiveDataset, FeaturesDataset
from tqdm import tqdm

def parse_args():
    parser = get_args_parser()
    parser.add_argument('--img_dir', type=str, default='',
                        help='Input image path')
    parser.add_argument('--out_dir', type=str, default='./output/',
                        help='Result directory path')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Sample size of SmoothGrad')
    parser.add_argument("--ckpt_path", type=str, help="Model Checkpoint Path")
    parser.add_argument("--probe_ckpt", type=str)
    args = parser.parse_args()
    return args

def get_finetune_saliency(img_name, device, args):
    img_name = '_'.join(args.img_dir.split('/')[-3:]).split('.')[0]
    if args.ckpt_path == None:
        model = timm.create_model(args.model_name, pretrained=True, num_classes=1)
    else:
        model = timm.create_model(args.model_name, pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(args.ckpt_path).state_dict())
    #model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)
    img = Image.open(args.img_dir).convert('RGB')
    preprocessed_img = torch.unsqueeze(transform(img), 0)
    model = model.to(device)
    smooth_grad = SmoothGrad(
        pretrained_model=model,
        cuda=True,
        n_samples=args.n_samples,
        magnitude=True)
    smooth_saliency = smooth_grad(preprocessed_img, index=None)
    #save_as_gray_image(smooth_saliency, os.path.join(args.out_dir, 'smooth_grad.jpg'))
    save_as_heatmap(smooth_saliency, os.path.join(args.out_dir, f'{args.model_name}_ft_{img_name}_grad.png'))
    img = cv2.resize(cv2.imread(args.img_dir, 1), smooth_saliency.shape[1:])
    save_as_overlay(img, smooth_saliency, os.path.join(args.out_dir, f'{args.model_name}_ft_{img_name}.png'))
    return smooth_saliency
    
def train_linear_probe(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device, args):
    best_acc_val = 0
    best_acc_train = 0
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
            train_acc = sum(epoch_acc)/float(len(epoch_acc))
            if val_acc > best_acc_val:
                torch.save(model, f'./logs/linear_probe_ckpts/{args.model_name}_{args.task}.ckpt')
                best_acc_val = val_acc
                best_acc_train = train_acc
    return best_acc_train, best_acc_val
    
def get_linear_probe_saliency(img_name, device, args):
    
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)
    

    if args.probe_ckpt is None:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_flip'), transform=transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True)
        train_features, train_labels = extract_features(model, train_loader, device)
        val_features, val_labels = extract_features(model, val_loader, device)
        train_feat_dataset = FeaturesDataset(train_features, train_labels)
        val_feat_dataset = FeaturesDataset(val_features, val_labels)
    
        train_feat_loader = DataLoader(train_feat_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
        val_feat_loader = DataLoader(val_feat_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        linear_model = LinearModel(train_features.shape[-1], args.num_classes, args.dropout_rate)
        linear_model = linear_model.to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(linear_model.parameters(), lr=args.learning_rate,
                                weight_decay=args.weight_decay, amsgrad=False)
        best_acc_train, best_acc_val = train_linear_probe(linear_model, train_feat_loader, val_feat_loader, criterion, optimizer, None, device, args)
        print("Best acc validation", best_acc_val)
        print("Best acc train", best_acc_train)
    else:
        linear_model = torch.load(args.probe_ckpt)
        #linear_model.load_state_dict(torch.load(args.probe_ckpt).state_dict())

    model.train()
    full_model = LinearProbeModel(model, linear_model)
    img = Image.open(args.img_dir).convert('RGB')
    preprocessed_img = torch.unsqueeze(transform(img), 0)
    smooth_grad = SmoothGrad(
        pretrained_model=full_model,
        cuda=True,
        n_samples=args.n_samples,
        magnitude=True)
    smooth_saliency = smooth_grad(preprocessed_img, index=None)
    #save_as_gray_image(smooth_saliency, os.path.join(args.out_dir, 'smooth_grad.jpg'))
    save_as_heatmap(smooth_saliency, os.path.join(args.out_dir, f'{args.model_name}_lp_{img_name}_grad.png'))
    img = cv2.resize(cv2.imread(args.img_dir, 1), smooth_saliency.shape[1:])
    save_as_overlay(img, smooth_saliency, os.path.join(args.out_dir, f'{args.model_name}_lp_{img_name}.png'))
    return smooth_saliency
    
    
def main():
    args = parse_args()
    device = torch.device(f'cuda:0')
    img_name = '_'.join(args.img_dir.split('/')[-3:]).split('.')[0]
    ft_map = get_finetune_saliency(img_name, device, args)
    lp_map = get_linear_probe_saliency(img_name, device, args)
    img = cv2.resize(cv2.imread(args.img_dir, 1), ft_map.shape[1:])
    save_diff_map(ft_map, lp_map, img, os.path.join(args.out_dir, f'{args.model_name}_diff_{img_name}_grad.png'),os.path.join(args.out_dir,  f'{args.model_name}_diff_{img_name}.png'))
if __name__ == "__main__":
    main()
    
    