import torch
from dataload.retinexDCEloader import retinexDCE_loader
from torchvision import transforms
from torch.utils.data import DataLoader
from network.retinex_dce import SimpleRetinexDce, DecomposeNet
import torch.optim as optim
import torch.nn as nn
import os
from dataload import zeroDCEloader
from torchvision.utils import save_image
from loss import PerceptualLoss, VGGLoss, CharbonnierLoss, CombinedLoss, RetinexLoss, CombinedLoss1, ColorLoss
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
# from transformers import AutoModelForSequenceClassification
# from accelerate import Accelerator
# from bitsandbytes import quantize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import cv2
import torch.nn.functional as F
import loss1
import numpy as np
from sklearn.model_selection import train_test_split

def train_decom():
    save_dir = "./train_oneByOne/traindecompose"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    dataset = retinexDCE_loader("Train_data/lol_dataset2/our485/")
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
    
    model = DecomposeNet()
    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    num_epochs = 100
    
    criterion = VGGLoss()
    criterion1 = CharbonnierLoss()
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
        
            R_low, L_low = model(low_light_imgs)
            R_high, L_high = model(well_lit_imgs)
            
           
            L_low_3 = torch.concat([L_low, L_low, L_low], dim=1)
            low_out = R_low*L_low_3
            
            L_high_3 = torch.concat([L_high, L_high, L_high], dim=1)
            normal_out = R_high*L_high_3
            

            loss_vgg_high = criterion(normal_out, well_lit_imgs)
            loss_vgg_low = criterion(low_out, low_light_imgs)
            
            recon_loss_mutual_low = F.l1_loss(R_high * L_low_3, low_light_imgs)
            recon_loss_mutual_high = F.l1_loss(R_low * L_high_3, well_lit_imgs)
            loss_charon_r = criterion1(R_high, R_low)
            
            loss = 0.0001*loss_vgg_low + 0.0001*loss_vgg_high + 0.02*recon_loss_mutual_low + 0.02*recon_loss_mutual_high + loss_charon_r
            
            loss.backward()
            optimizer.step()
            
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
                # optimizer.zero_grad()
            
                R_low, L_low = model(low_light_imgs)
                R_high, L_high = model(well_lit_imgs)
                
            
                L_low_3 = torch.concat([L_low, L_low, L_low], dim=1)
                low_out = R_low*L_low_3
                
                L_high_3 = torch.concat([L_high, L_high, L_high], dim=1)
                normal_out = R_high*L_high_3
                

                loss_vgg_high = criterion(normal_out, well_lit_imgs)
                loss_vgg_low = criterion(low_out, low_light_imgs)
                recon_loss_mutual_low = F.l1_loss(R_high * L_low_3, low_light_imgs)
                recon_loss_mutual_high = F.l1_loss(R_low * L_high_3, well_lit_imgs)
                loss_charon_r = criterion1(R_high, R_low)
                
                val_loss += 0.0001*loss_vgg_low + 0.0001*loss_vgg_high + 0.02*recon_loss_mutual_low + 0.02*recon_loss_mutual_high + loss_charon_r
        avg_val_loss = val_loss / len(val_dataloader)
            # loss.backward()
            # optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss}, Validation_loss: {avg_val_loss}')
            
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_dir, f"train_epoch_{epoch}_batch_{i}_RLow.jpg")
            save_image(R_low, save_path, normalize=True)
            save_path1 = os.path.join(save_dir, f"train_epoch_{epoch}_batch_{i}_RHigh.jpg")
            save_image(R_high, save_path1, normalize=True)
            
            save_path2 = os.path.join(save_dir, f"train_epoch_{epoch}_batch_{i}_iLow.jpg")
            save_image(L_low, save_path2, normalize=True)
            save_path3 = os.path.join(save_dir, f"train_epoch_{epoch}_batch_{i}_iHigh.jpg")
            save_image(L_high, save_path3, normalize=True)
            
            save_path4 = os.path.join(save_dir, f"train_epoch_{epoch}_batch_{i}_wholeLow.jpg")
            save_image(low_out, save_path4, normalize=True)
            save_path5 = os.path.join(save_dir, f"train_epoch_{epoch}_batch_{i}_wholeHigh.jpg")
            save_image(normal_out, save_path5, normalize=True)
            
        if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"./weights/best_traindecompose.pth")
        

if __name__ == "__main__":
    train_decom()
                    
            
        
            
            
            
    