import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import cv2

class SIDDataset(Dataset):
    def __init__(self, long_dir, short_dir, transform=None):
        self.long_images = sorted(glob.glob(os.path.join(long_dir, '*/*.npy')))
        self.short_images = sorted(glob.glob(os.path.join(short_dir, '*/*.npy')))
        self.transform = transform
        self.new_size = (224, 224)

    def __getitem__(self, index):
        long_path = self.long_images[index]
        short_path = self.short_images[index]

        # Load numpy arrays
        long_image = np.load(long_path).astype(np.float32)
        short_image = np.load(short_path).astype(np.float32)
        
        # Resize images
        long_image = cv2.resize(long_image, self.new_size)
        short_image = cv2.resize(short_image, self.new_size)

        # Normalize if necessary, for example: 
        # long_image /= 255.0
        # short_image /= 255.0

        # Convert numpy arrays to torch tensors and permute dimensions
        long_tensor = torch.from_numpy(long_image).permute(2, 0, 1)
        short_tensor = torch.from_numpy(short_image).permute(2, 0, 1)

        if self.transform:
            long_tensor = self.transform(long_tensor)
            short_tensor = self.transform(short_tensor)
        # print(short_tensor.shape)
        return short_tensor, long_tensor


    def __len__(self):
        return len(self.long_images)
