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
import optuna


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # Initialize Convolutional layers
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm layers
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        # Initialize Linear layers
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def train(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    save_dir = "./train_experiment/4channelrefinenodynamicdenoisebatch8UnetLOLSynaddscheduler"
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
    L_TV = loss1.L_TV()

    model = RetinexUnet()
    # model.apply(weights_init)

    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    optimizer = optim.AdamW(model.parameters(), lr=lr) 
    num_epochs = 80
    # best_train_loss = float('inf')
    criterion = VGGLoss()
    criterion1 = CharbonnierLoss()
    # criterion2 = ColorLoss()
    best_val_loss = float('inf')
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        model.train()
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            
            R_low, R_high, I_low, I_high = model(low_light_imgs, well_lit_imgs)
            I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
            I_high_3 = torch.concat([I_high, I_high, I_high], dim=1)
            low_output = R_low*I_low_3
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
            

            #add loss
            loss = loss_spa + loss_col + loss_exp + 0.01*recon_loss_mutual_low + 0.01*recon_loss_mutual_high + loss_charon + loss_r + loss_i +0.02*loss_vgg1 + loss_charon1
            # + 0.5*loss_vgg1 + 0.5*loss_charon1 + 10 * loss_color2
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            # if (epoch + 1) % 5 == 0:
            #     save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
            #     save_image(low_output, save_path, normalize=True)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Validation")):
                with torch.no_grad():
                    low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
                    R_low, R_high, I_low, I_high = model(low_light_imgs, well_lit_imgs)
                    I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
                    I_high_3 = torch.concat([I_high, I_high, I_high], dim=1)
                    low_output = R_low*I_low_3
                    
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
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss}, Validation Loss: {avg_val_loss.item()}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"./weights/4channelrefinenodynamicdenoisebatch8UnetLOLSynaddscheduler.pth")
    return val_loss


def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, (low_light_imgs, well_lit_imgs) in enumerate(dataloader):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            R_low, R_high, I_low, I_high = model(low_light_imgs, well_lit_imgs)
            I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
            low_output = R_low*I_low_3
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            save_path1 = os.path.join(save_dir, f"test_batch_{i+1}_truth.jpg")
            save_image(low_output, save_path, normalize=True)
            save_image(well_lit_imgs, save_path1, normalize=True)
            
def objective(trial):
    # Load dataset
    dataset = retinexDCE_loader("Train_data/VE-LOL-L-Syn/train/")
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train(trial)

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Train the model with the best hyperparameters
    best_lr = study.best_params['lr']
    print("Best LR:", best_lr)
    # train()
    
    # test_dataset = retinexDCE_loader("Train_data/VE-LOL-L-Syn/test/")
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # model = RetinexUnet()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # state_dict = torch.load("./weights/4channelrefinenodynamicdenoisebatch8UnetLOLSynaddscheduler.pth")

    # # Create a new state dictionary with the "module." prefix removed from each key
    # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(new_state_dict)  # Load the trained weights
    # model.to(device)
    # save_dir = "./Test_image/4channelrefinenodynamicdenoisebatch8UnetLOLSynaddscheduler"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # test_model(model, test_dataloader, device, save_dir)