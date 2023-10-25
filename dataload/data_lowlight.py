import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class LowLightDataset(Dataset):
    def __init__(self, dataset_path):
        self.low_light_paths = sorted(glob(os.path.join(dataset_path, 'low', '*')))
        self.well_lit_paths = sorted(glob(os.path.join(dataset_path, 'high', '*')))

        # Check if filenames match
        assert all([os.path.basename(low) == os.path.basename(high) for low, high in zip(self.low_light_paths, self.well_lit_paths)]), "Mismatched filenames between low and high directories."

        # Compute mean and std for normalization
        self.mean, self.std = self.compute_mean_std()

        # Data augmentation and transformations
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def compute_mean_std(self):
    # Compute mean and std for the dataset for normalization
        resize_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        all_images = [Image.open(p).convert('RGB') for p in self.low_light_paths + self.well_lit_paths]
        all_tensors = [resize_transform(img) for img in all_images]
        stacked_images = torch.stack(all_tensors, dim=0)
        mean = torch.mean(stacked_images, dim=[0, 2, 3])
        std = torch.std(stacked_images, dim=[0, 2, 3])
        return mean.tolist(), std.tolist()



    def __len__(self):
        return len(self.low_light_paths)

    def __getitem__(self, idx):
        low_light_img = Image.open(self.low_light_paths[idx]).convert('RGB')
        well_lit_img = Image.open(self.well_lit_paths[idx]).convert('RGB')

        # Ensure that the same transform is applied to both images
        seed = np.random.randint(2147483647)
        np.random.seed(seed)
        low_light_img = self.transform(low_light_img)
        np.random.seed(seed)
        well_lit_img = self.transform(well_lit_img)

        return low_light_img, well_lit_img
