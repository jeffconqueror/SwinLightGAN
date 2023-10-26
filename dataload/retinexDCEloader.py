import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_low_train_list(lowlight_images_path):
    # print("Searching in:", lowlight_images_path + "low/*.png")
    image_list_lowlight = glob.glob(lowlight_images_path + "low/*.png")

    # print("Detected images:", image_list_lowlight)
    # print("Loaded images:", image_list_lowlight)
    train_list = image_list_lowlight
    random.shuffle(train_list)
    return train_list

def populate_high_train_list(lowlight_images_path):
    # print("Searching in:", lowlight_images_path + "low/*.png")
    image_list_lowlight = glob.glob(lowlight_images_path + "high/*.png")

    # print("Detected images:", image_list_lowlight)
    # print("Loaded images:", image_list_lowlight)
    train_list = image_list_lowlight
    random.shuffle(train_list)
    return train_list

	

class retinexDCE_loader(data.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_low_list = populate_low_train_list(lowlight_images_path) 
		self.train_high_list = populate_high_train_list(lowlight_images_path)
		self.size = 400

		self.data_list = self.train_low_list
		print("Total training examples:", len(self.train_low_list))


		

	def __getitem__(self, index):
        #low
		data_lowlight_path = self.data_list[index]
		
		data_lowlight = Image.open(data_lowlight_path)
		
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.LANCZOS)

		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()
		
		#high
		data_highlight_path = self.data_list[index]
		
		data_highlight = Image.open(data_highlight_path)
		
		data_highlight = data_highlight.resize((self.size,self.size), Image.LANCZOS)

		data_highlight = (np.asarray(data_highlight)/255.0) 
		data_highlight = torch.from_numpy(data_highlight).float()


		return data_lowlight.permute(2,0,1), data_highlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)