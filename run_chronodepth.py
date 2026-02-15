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
from torch.utils.data import Subset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

from src.perspective_data import PerspectiveDataset, FeaturesDataset
from src.models import LinearModel, LinearModelMulti
from src.utils import binary_accuracy, accuracy, CosineAnnealingWithWarmup, get_args_parser, get_transform_wo_crop, get_foundation_model
import json
from submodule.ChronoDepth.chronodepth import ChronoDepthPipeline
from submodule.ChronoDepth.chronodepth.unet_chronodepth import DiffusersUNetSpatioTemporalConditionModelChronodepth
from submodule.ChronoDepth.src.utils.video_utils import resize_max_res

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)  # (image_tensor, label, path)


@torch.no_grad()
def extract_features_chrono_from_loader(model, data_loader, device):
    model.to(device)
    features, labels_list = [], []

    for data in tqdm(data_loader):
        _, labels, paths = data  # skip transformed tensor

        for i in range(len(paths)):
            img_path = paths[i]
            label = labels[i].item()

            im = cv2.imread(img_path)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = np.expand_dims(im, axis=0)
            im = resize_max_res(im, max_res=518)
            im = im.astype(np.float32) / 255.0

            out = model(
                im,
                num_inference_steps=5,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.0,
                generator=torch.Generator(device=device).manual_seed(1234),
                infer_mode='ours',
                sigma_epsilon=-4.0,
                return_latents=True
            )

            _, depth_latents = out
            latent = depth_latents.squeeze(0).mean(dim=(1, 2, 3))
            features.append(latent.detach().cpu())
            labels_list.append(label)

    features = torch.stack(features).float()
    labels = torch.tensor(labels_list).float()

    return features, labels




def extract_features_dpt(device, split, args):
    model, transform = get_foundation_model(args.model_name, args)
    csv_name = split + f'_{args.task}_balanced.csv'
    # if split == 'train':
    #     if args.flip:
    #         data_path = os.path.join(args.data_dir, 'train_flip')
    #     else:
    #         data_path = os.path.join(args.data_dir, 'train')
    if split == 'train':
        data_path = os.path.join(args.data_dir, 'train')
    elif split == 'val':
        data_path = os.path.join(args.data_dir, 'train')
    else:
        data_path = os.path.join(args.data_dir, 'test')
        
    label_csv = os.path.join(args.data_dir, csv_name)
    img_labels = pd.read_csv(label_csv).to_numpy()
    features = []
    labels_list = []
    for idx in tqdm(range(len(img_labels))):
        img_path = os.path.join(data_path, img_labels[idx, 0])
        labels = img_labels[idx, 1]
        raw_image = cv2.imread(img_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2RGB) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(image)
            feature = feature[3][0]
            feature = torch.mean(feature, 1).detach().cpu()
        features.append(feature)
        labels_list.append(labels)
    features = torch.cat(features)
    labels = torch.Tensor(labels_list).squeeze()
    return features, labels
    
def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels_list = []
    for data in tqdm(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(images)
            if len(preds.shape)>3:
                preds = torch.mean(preds, (2, 3))
                #preds = torch.flatten(preds, start_dim=1)
            elif len(preds.shape)>2:
                preds = torch.mean(preds, 2)
            features.append(preds.cpu())
            labels_list.append(labels.cpu())
    features = torch.cat(features)
    labels = torch.cat(labels_list).squeeze()
    return features, labels

def train_linear_probe(model, train_loader, test_loader, val_loader, human_loader, criterion, optimizer, lr_scheduler, device, args):
    best_acc_val = 0
    best_acc_train = 0
    best_acc_human = 0
    best_acc_test = 0
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
            #lr_scheduler.step()
            #acc = accuracy(preds,labels)[0]
            acc = binary_accuracy(preds, labels)
            epoch_acc.append(acc)
            epoch_loss.append(loss.item())
        with torch.no_grad():
            test_acc, test_loss = evaluate_linear_probe(model, test_loader, criterion, device, args)
            val_acc, val_loss = evaluate_linear_probe(model, val_loader, criterion, device, args)
            # human_acc, human_loss = evaluate_linear_probe(model, human_loader, criterion, device, args)
            human_acc, human_loss = 0, 0
            train_acc = sum(epoch_acc)/float(len(epoch_acc))
            if val_acc > best_acc_val:
                best_acc_val = val_acc
                best_acc_train = train_acc
                best_acc_test = test_acc
                # best_acc_human = human_acc
        if args.wandb:
            wandb.log({'train_acc':sum(epoch_acc)/float(len(epoch_acc)), 'train_loss':sum(epoch_loss)/float(len(epoch_loss)),
                    'val_acc':val_acc, 'val_loss':val_loss, 'human_acc':human_acc, 'human_loss':human_loss, 'test_acc':test_acc, 'test_loss':test_loss})
    return best_acc_train, best_acc_test, best_acc_val, best_acc_human
    
def evaluate_linear_probe(model, data_loader, criterion, device, args):
    model.eval()
    epoch_loss = []
    preds_list = []
    labels_list = []
    for i, batch in enumerate(data_loader):
        features, labels = batch
        features = features.to(device)
        labels = labels.float().to(device)
        labels = torch.unsqueeze(labels, 1)
        preds = model(features)
        loss = criterion(preds, labels)
        preds_list.append(preds)
        labels_list.append(labels)
        epoch_loss.append(loss.item())
    preds = torch.cat(preds_list).squeeze()
    labels = torch.cat(labels_list).squeeze()
    epoch_acc = binary_accuracy(preds, labels)
    return epoch_acc, sum(epoch_loss)/float(len(epoch_loss))

def run_extract_features(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    if args.model_name in timm.list_models(pretrained=True):
        model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        transform = get_transform_wo_crop(data_config)

    else:
        model = get_foundation_model(args.model_name, args)
        transform = Compose([
                        ToTensor(),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
                        Resize(512, interpolation=InterpolationMode.NEAREST)])


                        
    if not args.flip:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform)
    else:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_flip'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform)
    # human_dataset = PerspectiveDataset(Path(args.data_dir), transforms=transform, split='human', task=args.task)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    # human_loader = DataLoader(human_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    model = model.to(device)

    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    # human_features, human_labels = extract_features(model, human_loader, device)

    # return train_features, train_labels, test_features, test_labels, val_features, val_labels, human_features, human_labels
    return train_features, train_labels, test_features, test_labels, val_features, val_labels, None, None

def run_linear_probe(args):
    if args.wandb:
        wandb_run = wandb.init(project='gs-perception-linear-probe', 
                            config={                   
                                "learning_rate": args.learning_rate,
                                "architecture": args.model_name,
                                "epochs": args.epochs,
                                } )
    device = torch.device(f'cuda:{args.gpu_id}')
    
    if args.model_name == 'depth_anything':
        train_features, train_labels = extract_features_dpt(device, 'train', args)
        test_features, test_labels = extract_features_dpt(device, 'test', args)
        val_features, val_labels = extract_features_dpt(device, 'val', args)
        # human_features, human_labels = extract_features_dpt(device, 'human', args)

    elif args.model_name == 'chronodepth':
        # Load ChronoDepth model
        unet = DiffusersUNetSpatioTemporalConditionModelChronodepth.from_pretrained(
            "jhshao/ChronoDepth-v1",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        chrono_model = ChronoDepthPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        chrono_model.n_tokens = 10
        chrono_model.chunk_size = 5
        chrono_model.to(device)

        # Create PerspectiveDataset loaders
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
            Resize(518, interpolation=InterpolationMode.BILINEAR)
        ])
        train_dataset = ImageFolderWithPaths(os.path.join(args.data_dir, 'train'), transform=transform)
        test_dataset = ImageFolderWithPaths(os.path.join(args.data_dir, 'test'), transform=transform)
        val_dataset = ImageFolderWithPaths(os.path.join(args.data_dir, 'val'), transform=transform)
        # human_dataset = PerspectiveDataset(Path(args.data_dir), transforms=transform, split='human', task=args.task, return_path=True)
        print("Train size:", len(train_dataset))
        print("Test size:", len(test_dataset))
        print("Val size:", len(val_dataset))
            
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        # human_loader = DataLoader(human_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        args.device = device

        train_features, train_labels = extract_features_chrono_from_loader(chrono_model, train_loader, device)
        test_features, test_labels   = extract_features_chrono_from_loader(chrono_model, test_loader, device)
        val_features, val_labels     = extract_features_chrono_from_loader(chrono_model, val_loader, device)
        # human_features, human_labels = extract_features_chrono_from_loader(chrono_model, human_loader, device)

    else:
        train_features, train_labels, test_features, test_labels, val_features, val_labels, human_features, human_labels = run_extract_features(args)

    # Prepare datasets and loaders
    train_feat_dataset = FeaturesDataset(train_features, train_labels)
    test_feat_dataset  = FeaturesDataset(test_features, test_labels)
    val_feat_dataset   = FeaturesDataset(val_features, val_labels)
    # human_feat_dataset = FeaturesDataset(human_features, human_labels)

    train_feat_loader = DataLoader(train_feat_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    test_feat_loader  = DataLoader(test_feat_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_feat_loader   = DataLoader(val_feat_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # human_feat_loader = DataLoader(human_feat_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    human_feat_loader = None

    # Linear classifier training
    criterion = torch.nn.BCEWithLogitsLoss()
    linear_model = LinearModel(train_features.shape[-1], args.num_classes, args.dropout_rate).to(device)
    optimizer = torch.optim.AdamW(linear_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=False)
    lr_scheduler = None

    best_acc_train, best_acc_test, best_acc_val, best_acc_human = train_linear_probe(
        linear_model,
        train_feat_loader,
        test_feat_loader,
        val_feat_loader,
        human_feat_loader,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        args
    )

    print("Best acc validation", best_acc_val)
    print("Best acc test", best_acc_test)
    print("Best acc train", best_acc_train)
    print("Best acc human", best_acc_human)

    log_file = f'/users/jfeng39/VPT/logs/{args.task}_results_{args.split}.json'
    with open(log_file, 'r') as f:
        results = json.load(f)
    results[args.model_name] = [best_acc_train, best_acc_test, best_acc_val, best_acc_human]
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)
        print('Wrote results to', log_file)

    if args.wandb:
        wandb_run.finish()

        
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run_linear_probe(args)
