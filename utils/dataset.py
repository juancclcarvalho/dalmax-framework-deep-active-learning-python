import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image


import torch.backends.cudnn as cudnn    

class Dataset():
    def __init__(self, data_dir, batch_size, img_size=128):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        print("\nConstructing Dataset...")
        print("Dataset: ", data_dir)
        print("Image size: ", img_size)
        print("Batch size: ", batch_size)
        print("\n")

        self.labels = []
        self.classes = []

        self.verify()
        self.train_loader, self.test_loader, self.num_classes = self.load_data()

        self.unlabelled_loader = None

    @staticmethod
    def load_image_rgb(image_path):
        image = Image.open(image_path).convert("RGB")
        # retornar a imagem e suas dimensões
        return image, image.size
    
    def verify(self):
        data_dir = self.data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"Directory {data_dir} not found.")
        if not os.path.exists(os.path.join(data_dir, "train")):
            raise Exception(f"Directory {os.path.join(data_dir, 'train')} not found.")
        if not os.path.exists(os.path.join(data_dir, "test")):
            raise Exception(f"Directory {os.path.join(data_dir, 'test')} not found.")
        
        # batch_size > 0
        if self.batch_size <= 0:
            raise Exception(f"Batch size must be greater than 0.")
        
        # img_size > 0
        if self.img_size <= 0:
            raise Exception(f"Image size must be greater than 0.")

    def load_data(self):
        img_size = self.img_size
        data_dir = self.data_dir
        batch_size = self.batch_size
        
        # Transforms
        transform1 = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Aplicar 6 técnicas de aumento de dados
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.8, 1.2))
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, len(train_dataset.classes)
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_name_classes(self):
        self.classes = self.train_loader.dataset.classes
        return self.classes
    
    def get_data_dir(self):
        return self.data_dir