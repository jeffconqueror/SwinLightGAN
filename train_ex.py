import torch
from dataload.retinexDCEloader import retinexDCE_loader
from torchvision import transforms
from torch.utils.data import DataLoader
from network.retinex_dce import RetinexUnet
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
import matplotlib.pyplot as plt


def weights_init(m, negative_slope=0.01):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=negative_slope)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=negative_slope)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)




def train():
    save_dir = "./train_experiment/LOLSyndeeperResdenoisereducelayerscheduleradd485reducedecomSE"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    dataset = retinexDCE_loader("Train_data/VE-LOL-L-Syn/train/")
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)

    L_color = loss1.L_color()
    L_spa = loss1.L_spa()
    L_exp = loss1.L_exp(16,0.6)
    model = RetinexUnet()
    # model.apply(weights_init)

    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    num_epochs = 120
    # best_train_loss = float('inf')
    criterion = VGGLoss()
    criterion1 = CharbonnierLoss()
    # criterion2 = ColorLoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        train_loss_components = {
            'loss_spa': 0.0,
            'loss_col': 0.0,
            'loss_exp': 0.0,
            'loss_charon': 0.0,
            'recon_loss_mutual_low': 0.0,
            'recon_loss_mutual_high': 0.0,
            'loss_r': 0.0,
            'loss_i': 0.0,
            'loss_vgg1': 0.0,
            'loss_charon1': 0.0
        }
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            
            R_low, R_high, I_low, I_high, low_output = model(low_light_imgs, well_lit_imgs)
            I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
            I_high_3 = torch.concat([I_high, I_high, I_high], dim=1)
            # low_output = R_low*I_low_3
            #loss for enhancer
            # Loss_TV = 200*L_TV(A)
            normal_output = R_high*torch.concat([I_high, I_high, I_high], dim=1)
            loss_spa = 5*torch.mean(L_spa(low_output, low_light_imgs))
            loss_col = 50*torch.mean(L_color(low_output))
            loss_exp = 100*torch.mean(L_exp(low_output))

            #loss for R * I
            # loss_vgg = criterion(normal_output, well_lit_imgs)
            loss_charon = criterion1(normal_output, well_lit_imgs)
            # loss_color1 = criterion2(normal_output, well_lit_imgs)
            recon_loss_mutual_low = F.l1_loss(R_high * I_low_3, low_light_imgs)
            recon_loss_mutual_high = F.l1_loss(R_low * I_high_3, well_lit_imgs)
            
            #loss for R_low/R_high, I_low/I_high
            loss_r = criterion1(R_low, R_high)
            loss_i = criterion1(I_low, I_high)
            
            #loss for enhanced
            loss_vgg1 = criterion(low_output, well_lit_imgs) #vgg loss
            loss_charon1 = criterion1(low_output, well_lit_imgs) #CharbonnierLoss
            # loss_color2 = criterion2(low_output, well_lit_imgs) #color loss
            
            train_loss_components['loss_spa'] += loss_spa.item()
            train_loss_components['loss_col'] += loss_col.item()
            train_loss_components['loss_exp'] += loss_exp.item()
            train_loss_components['loss_charon'] += loss_charon.item()
            train_loss_components['recon_loss_mutual_low'] += 0.01*recon_loss_mutual_low.item()
            train_loss_components['recon_loss_mutual_high'] += 0.01*recon_loss_mutual_high.item()
            train_loss_components['loss_r'] += loss_r.item()
            train_loss_components['loss_i'] += loss_i.item()
            train_loss_components['loss_vgg1'] += 0.02*loss_vgg1.item()
            train_loss_components['loss_charon1'] += loss_charon1.item()
            #add loss
            loss = loss_spa + loss_col + loss_exp + 0.01*recon_loss_mutual_low + 0.01*recon_loss_mutual_high + loss_charon + loss_r + loss_i +0.02*loss_vgg1 + loss_charon1
            train_loss += loss.item()
            # + 0.5*loss_vgg1 + 0.5*loss_charon1 + 10 * loss_color2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # if (epoch + 1) % 5 == 0:
            #     save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
            #     save_image(low_output, save_path, normalize=True)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Validation")):
                with torch.no_grad():
                    low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
                    R_low, R_high, I_low, I_high, low_output = model(low_light_imgs, well_lit_imgs)
                    I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
                    I_high_3 = torch.concat([I_high, I_high, I_high], dim=1)
                    # low_output = R_low*I_low_3
                    
                    normal_output = R_high*torch.concat([I_high, I_high, I_high], dim=1)
                    loss_spa = 5*torch.mean(L_spa(low_output, low_light_imgs))
                    loss_col = 5*torch.mean(L_color(low_output))
                    loss_exp = 100*torch.mean(L_exp(low_output))

                    #loss for R * I
                    # loss_vgg = criterion(normal_output, well_lit_imgs) #vgg loss
                    loss_charon = criterion1(normal_output, well_lit_imgs)
                    recon_loss_mutual_low = F.l1_loss(R_high * I_low_3, low_light_imgs)
                    recon_loss_mutual_high = F.l1_loss(R_low * I_high_3, well_lit_imgs)
                    
                    loss_r = criterion1(R_low, R_high)
                    loss_i = criterion1(I_low, I_high)
            
                    loss_vgg1 = criterion(low_output, well_lit_imgs) #vgg loss
                    loss_charon1 = criterion1(low_output, well_lit_imgs)
                    
                    val_loss +=  loss_spa + loss_col + loss_exp + 0.01*recon_loss_mutual_low + 0.01*recon_loss_mutual_high + loss_charon + loss_r + loss_i +0.02*loss_vgg1 + loss_charon1
                    
                    if (epoch + 1) % 5 == 0:
                        save_path = os.path.join(save_dir, f"val_epoch_{epoch}_batch_{i}.jpg")
                        save_image(low_output, save_path, normalize=True)
                        # save_path1 = os.path.join(save_dir, f"val_epoch_{epoch}_batch_{i}_high.jpg")
                        # save_image(well_lit_imgs, save_path1, normalize=True)
            avg_val_loss = val_loss / len(val_dataloader)
            avg_train_loss = train_loss / len(train_loader)
            if isinstance(avg_train_loss, torch.Tensor):
                avg_train_loss = avg_train_loss.detach().cpu().item()
            if isinstance(avg_val_loss, torch.Tensor):
                avg_val_loss = avg_val_loss.detach().cpu().item()

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss}, Validation Loss: {avg_val_loss}')
            # num_batches = len(train_loader)
            # for k in train_loss_components.keys():
            #     train_loss_components[k] /= num_batches
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss Components:")
            # for loss_name, loss_value in train_loss_components.items():
            #     print(f"{loss_name}: {loss_value}")
            
            # scheduler.step()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"./weights/LOLSyndeeperResdenoisereducelayerscheduleradd485reducedecomSE.pth")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./train_experiment/LOLSyndeeperResdenoisereducelayerscheduleradd485reducedecomSE/training_validation_loss_plot.png')


def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, (low_light_imgs, well_lit_imgs) in enumerate(dataloader):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            _, _, _, _, low_output = model(low_light_imgs, well_lit_imgs)
            # I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
            # low_output = R_low*I_low_3
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            save_path1 = os.path.join(save_dir, f"test_batch_{i+1}_truth.jpg")
            save_image(low_output, save_path, normalize=True)
            save_image(well_lit_imgs, save_path1, normalize=True)

if __name__ == "__main__":
    train()
    
    test_dataset = retinexDCE_loader("Train_data/VE-LOL-L-Syn/test/")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    model = RetinexUnet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("./weights/LOLSyndeeperResdenoisereducelayerscheduleradd485reducedecomSE.pth")

    # Create a new state dictionary with the "module." prefix removed from each key
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)  # Load the trained weights
    model.to(device)
    save_dir = "./Test_image/LOLSyndeeperResdenoisereducelayerscheduleradd485reducedecomSE"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(model, test_dataloader, device, save_dir)