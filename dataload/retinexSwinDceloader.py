import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)

class RetinexSwinDCELoader(data.Dataset):
    def __init__(self, images_path, size=224): #224*2
        self.lowlight_image_paths = sorted(glob.glob(os.path.join(images_path, "low", "*.png")))
        self.highlight_image_paths = sorted(glob.glob(os.path.join(images_path, "high", "*.png")))

        # Ensure the number of lowlight and highlight images is the same
        assert len(self.lowlight_image_paths) == len(self.highlight_image_paths), \
            "Mismatch in number of lowlight and highlight images!"

        self.size = size

    def __len__(self):
        return len(self.lowlight_image_paths)

    def load_image(self, path):
        img = Image.open(path)
        img = img.resize((self.size, self.size), Image.LANCZOS)
        img = np.asarray(img) / 255.0
        img = torch.from_numpy(img).float()
        return img.permute(2, 0, 1)

    def __getitem__(self, index):
        lowlight_img = self.load_image(self.lowlight_image_paths[index])
        highlight_img = self.load_image(self.highlight_image_paths[index])
        return lowlight_img, highlight_img


