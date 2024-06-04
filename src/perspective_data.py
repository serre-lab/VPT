import torch
from torch.utils.data import Dataset
import csv
import pandas as pd
from PIL import Image
import os
import numpy as np
import torchvision.datasets as datasets

class ViewDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_folder = datasets.ImageFolder(data_path, transform=transform)
    def __len__(self):
        return len(self.image_folder)
    def __getitem__(self, idx):
        path, _ = self.image_folder.imgs[idx]
        sample, target = self.image_folder.__getitem__(idx)
        return sample, target, path
    
    
class PerspectiveDatasetImageFolder(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_folder = datasets.ImageFolder(data_path, transform=transform)
    def __len__(self):
        return len(self.image_folder)
    def __getitem__(self, idx):
        path, _ = self.image_folder.imgs[idx]
        sample, target = self.image_folder.__getitem__(idx)
        return sample, target, path
    
class PerspectiveDataset(Dataset):
    def __init__(self, data_path, transforms=None, split='train', task='perspective', return_path=False):
        assert split in ['train', 'test', 'human', 'val']
        assert task in ['perspective', 'depth']
        csv_name = split + f'_{task}_balanced.csv'
        print(csv_name)
        label_csv = os.path.join(data_path, csv_name)
        if split == 'train':
            self.data_dir = os.path.join(data_path, 'train')
        elif split == 'val':
            self.data_dir = os.path.join(data_path, 'train')
        else:
            self.data_dir = os.path.join(data_path, 'test')
            
        self.img_labels = pd.read_csv(label_csv).to_numpy()
        self.transforms = transforms
        self.return_path = return_path
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_labels[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx, 1]
        if self.transforms:
            image = self.transforms(image)
        if self.return_path:
            return image, label, img_path
        return image, label


class FeaturesDataset(Dataset):
    def __init__(self, features, labels, img_paths=None):
        # m = torch.mean(features, axis=0)
        # s = torch.std(features, axis=0)
        # features -= m
        # features /= s
        self.data = features
        self.labels = labels
        self.img_paths = img_paths
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.img_paths is None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx], self.labels[idx], self.img_paths[idx]