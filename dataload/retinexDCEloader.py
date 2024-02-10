import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
from torchvision import transforms
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

random.seed(1143)


def populate_low_train_list(lowlight_images_path):
    # print("Searching in:", lowlight_images_path + "low/*.png")
    image_list_lowlight = glob.glob(lowlight_images_path + "low/*.png")

    # print("Detected images:", image_list_lowlight)
    # print("Loaded images:", image_list_lowlight)
    train_list = image_list_lowlight
    # random.shuffle(train_list)
    return train_list

def populate_high_train_list(lowlight_images_path):
    # print("Searching in:", lowlight_images_path + "low/*.png")
    image_list_lowlight = glob.glob(lowlight_images_path + "high/*.png")

    # print("Detected images:", image_list_lowlight)
    # print("Loaded images:", image_list_lowlight)
    train_list = image_list_lowlight
    # random.shuffle(train_list)
    return train_list

	

class retinexDCE_loader_train(data.Dataset):
    def __init__(self, lowlight_images_path) -> None:
        low_list = populate_low_train_list(lowlight_images_path)
        high_list = populate_high_train_list(lowlight_images_path)
        
        # Pairing and shuffling
        self.paired_list = list(zip(low_list, high_list))
        random.shuffle(self.paired_list)
        self.size = 224
        print("Total training examples:", len(self.paired_list))
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.RandomCrop(self.size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    
    def __getitem__(self, index):
        data_lowlight_path, data_highlight_path = self.paired_list[index]
        
        # data_lowlight_path = 
        
        data_lowlight = Image.open(data_lowlight_path)
        data_highlight = Image.open(data_highlight_path)
        # seed = torch.Generator().manual_seed(2147483647)
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.LANCZOS)
        data_lowlight = self.transform(data_lowlight)
        # data_lowlight = (np.asarray(data_lowlight)/255.0) 
        # data_lowlight = torch.from_numpy(data_lowlight).float()
		
		
		#high
        # data_highlight_path = self.data_highlight_path[index]
        
        data_highlight = data_highlight.resize((self.size,self.size), Image.LANCZOS)
        data_highlight = self.transform(data_highlight)
        # data_highlight = (np.asarray(data_highlight)/255.0) 
        # data_highlight = torch.from_numpy(data_highlight).float()
        
        
        return data_lowlight, data_highlight
       
    
    def __len__(self):
        return len(self.paired_list)
    
    
    
class retinexDCE_loader_test(data.Dataset):
    def __init__(self, lowlight_images_path) -> None:
        low_list = populate_low_train_list(lowlight_images_path)
        high_list = populate_high_train_list(lowlight_images_path)
        
        # Pairing and shuffling
        self.paired_list = list(zip(low_list, high_list))
        random.shuffle(self.paired_list)
        self.size = 224
        print("Total training examples:", len(self.paired_list))
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    
    def __getitem__(self, index):
        data_lowlight_path, data_highlight_path = self.paired_list[index]
        
        # data_lowlight_path = 
        data_lowlight = Image.open(data_lowlight_path)
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.LANCZOS)
        data_lowlight = self.transform(data_lowlight)
        # data_lowlight = (np.asarray(data_lowlight)/255.0) 
        # data_lowlight = torch.from_numpy(data_lowlight).float()
        
        
        #high
        # data_highlight_path = self.data_highlight_path[index]
        data_highlight = Image.open(data_highlight_path)
        data_highlight = data_highlight.resize((self.size,self.size), Image.LANCZOS)
        data_highlight = self.transform(data_highlight)
        # data_highlight = (np.asarray(data_highlight)/255.0) 
        # data_highlight = torch.from_numpy(data_highlight).float()
        
        
        return data_lowlight, data_highlight
    
    
    def __len__(self):
        return len(self.paired_list)





