import os
from glob import glob
from PIL import Image
# import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LowLightDataset(Dataset):
    def __init__(self, dataset_path):
        # Assuming your directories contain images with matching filenames
        self.low_light_paths = sorted(glob(os.path.join(dataset_path, 'low', '*')))
        self.well_lit_paths = sorted(glob(os.path.join(dataset_path, 'high', '*')))

        # Common transformations for SwinIR-based tasks
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # Resizing the image to 256x256 as an example
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] for RGB channels
        ])

    def __len__(self):
        # Assuming the number of low-light and well-lit images are the same
        return len(self.low_light_paths)

    def __getitem__(self, idx):
        low_light_img = Image.open(self.low_light_paths[idx]).convert('RGB') # Convert image to RGB
        well_lit_img = Image.open(self.well_lit_paths[idx]).convert('RGB')

        # Apply transformations
        low_light_img = self.transform(low_light_img)
        well_lit_img = self.transform(well_lit_img)

        return low_light_img, well_lit_img

        
        return low_light_img, well_lit_img
