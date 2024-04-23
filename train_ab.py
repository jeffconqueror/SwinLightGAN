import torch
from dataload.retinexDCEloader import retinexDCE_loader_train, retinexDCE_loader_test, UnpairedLowLightLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from network.retinex_dce import RetinexUnet
import torch.optim as optim
import torch.nn as nn
import os
# from dataload import zeroDCEloader
from torchvision.utils import save_image
from loss import PerceptualLoss, VGGLoss, CharbonnierLoss, CombinedLoss, RetinexLoss, CombinedLoss1, ColorLoss
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from transformers import AutoModelForSequenceClassification
# from accelerate import Accelerator
# from bitsandbytes import quantize
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torchsummary import summary
import warnings
import cv2
import torch.nn.functional as F
import loss1
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
from get_psnr_ssim import calculate_average_psnr_ssim


 
def prune_weights(model, threshold=1e-3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.abs(module.weight) > threshold
            module.weight.data *= mask.float()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.2)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.2)



def compute_loss(model, low_light_imgs, well_lit_imgs, loss_components, low_output):
    # low_output = model(low_light_imgs)
    loss_spa = 5 * torch.mean(loss_components['L_spa'](low_output, low_light_imgs))
    loss_col = 50 * torch.mean(loss_components['L_color'](low_output))
    loss_exp = 100 * torch.mean(loss_components['L_exp'](low_output))
    loss_vgg1 = loss_components['criterion'](low_output, well_lit_imgs) #vgg loss
    loss_charon1 = loss_components['criterion1'](low_output, well_lit_imgs) #CharbonnierLoss
    # Calculate the total loss
    return loss_spa + loss_col + loss_exp + 0.02 * loss_vgg1 + loss_charon1

initial_lr = 1e-8
max_lr = 2e-5

def lr_schedule(epoch, warmup_epochs=75, max_lr_epochs=600, total_epochs=750):
    if epoch < warmup_epochs:
        lr = initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
    elif epoch < max_lr_epochs:
        lr = max_lr
    else:
        lr = max_lr * (1.0 - (epoch - max_lr_epochs) / (total_epochs - max_lr_epochs))
    return lr
    
def train(model, train_loader, val_dataloader, device, save_dir, num_epochs=200):
    weights_dir_path = os.path.join("./weights", save_dir)
    if not os.path.exists(weights_dir_path):
        os.makedirs(weights_dir_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    L_color = loss1.L_color()
    L_spa = loss1.L_spa()
    L_exp = loss1.L_exp(16,0.6)

    model = nn.DataParallel(model)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) 
    # num_epochs = 150
    # best_train_loss = float('inf')
    criterion = VGGLoss()
    criterion1 = CharbonnierLoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    # plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            lr = lr_schedule(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            
            low_output = model(low_light_imgs)
            loss = compute_loss(model, low_light_imgs, well_lit_imgs, {
                'L_spa': L_spa,
                'L_color': L_color,
                'L_exp': L_exp,
                'criterion': criterion,
                'criterion1': criterion1
            }, low_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        ##validation part
        model.eval()
        val_loss = 0
        
        if (epoch + 1) % 5 == 0:
            val_loss = test(None, model, val_dataloader, device, save_dir, epoch=epoch, save=True)
        else:
            val_loss = test(None, model, val_dataloader, device, save_dir, epoch=epoch)    
        avg_val_loss = val_loss / len(val_dataloader)
        avg_train_loss = train_loss / len(train_loader)
        # plateau_scheduler.step(avg_val_loss)
        if isinstance(avg_train_loss, torch.Tensor):
            avg_train_loss = avg_train_loss.detach().cpu().item()
        if isinstance(avg_val_loss, torch.Tensor):
            avg_val_loss = avg_val_loss.detach().cpu().item()

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss}, Validation Loss: {avg_val_loss}')
        
        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            weights_file_path = os.path.join("./weights", save_dir, "model_epoch_{}.pth".format(epoch))
            torch.save(model.state_dict(), weights_file_path)

            
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plot_save_path = os.path.join(save_dir, "training_validation_loss_plot.png")
    plt.savefig(plot_save_path)

def test(args, model, test_loader, device, save_dir, scheduler = None, save=False, epoch=None):
    L_color = loss1.L_color()
    L_spa = loss1.L_spa()
    L_exp = loss1.L_exp(16,0.6)
    
    criterion = VGGLoss()
    criterion1 = CharbonnierLoss()
    val_loss = 0.0
    with torch.no_grad():
        for i, (low_light_imgs, well_lit_imgs) in enumerate(test_loader):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            low_output = model(low_light_imgs)
            
            # Calculate the different loss components
            loss = compute_loss(model, low_light_imgs, well_lit_imgs, {
                        'L_spa': L_spa,
                        'L_color': L_color,
                        'L_exp': L_exp,
                        'criterion': criterion,
                        'criterion1': criterion1
                    }, low_output)
            
            # Aggregate the total loss
            # loss = loss_spa + loss_col + loss_exp + 0.02 * loss_vgg1 + loss_charon1
            val_loss += loss.item()
            if save:
                save_path = os.path.join(save_dir, f"val_epoch_{epoch}_batch_{i+1}.jpg")
                save_image(low_output, save_path, normalize=True)

    return val_loss
 

    
def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, (low_light_imgs, well_lit_imgs) in enumerate(dataloader):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            low_output = model(low_light_imgs)
            # I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
            # low_output = R_low*I_low_3
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            save_path1 = os.path.join(save_dir, f"test_batch_{i+1}_truth.jpg")
            save_path2 = os.path.join(save_dir, f"test_batch_{i+1}_low.jpg")
            save_image(low_output, save_path, normalize=True)
            save_image(well_lit_imgs, save_path1, normalize=True)
            save_image(low_light_imgs, save_path2, normalize=True)
            
def test_real(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, low_light_imgs in enumerate(dataloader):
            low_light_imgs = low_light_imgs.to(device)
            low_output = model(low_light_imgs)
            # I_low_3 = torch.concat([I_low, I_low, I_low], dim=1)
            # low_output = R_low*I_low_3
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            # save_path1 = os.path.join(save_dir, f"test_batch_{i+1}_truth.jpg")
            save_path2 = os.path.join(save_dir, f"test_batch_{i+1}_low.jpg")
            save_image(low_output, save_path, normalize=True)
            # save_image(well_lit_imgs, save_path1, normalize=True)
            save_image(low_light_imgs, save_path2, normalize=True)

def best_weights(model, weights_folder, device, test_dataloader, save_dir):
    highest_psnr = 0
    best_weight_file = ''
    results_file = os.path.join(save_dir, "weights_evaluation_results.txt")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(results_file, "w") as f:
        for weight_file in os.listdir(weights_folder):
            print("using file: ", weight_file)
            if weight_file.endswith('.pth'):
                state_dict = torch.load(os.path.join(weights_folder, weight_file))
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
                model.to(device)
                test_model(model, test_dataloader, device, save_dir)
                average_psnr, average_ssim = calculate_average_psnr_ssim(100, save_dir)
                f.write(f"File: {weight_file}, PSNR: {average_psnr}, SSIM: {average_ssim}\n")
                if average_psnr > highest_psnr:
                    highest_psnr = average_psnr
                    best_weight_file = weight_file
        f.write(f"Best PSNR: {highest_psnr} using file {best_weight_file}\n")
    print("best psnr: ", highest_psnr)
    print("saved to ", best_weight_file)
    return best_weight_file, highest_psnr

if __name__ == "__main__":
    import torch.nn.utils.prune as prune
    model = RetinexUnet()
    
    save_dir = "./train_prune/LOLv2Syn_prune_ablation_noDenoise"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = retinexDCE_loader_train("Train_data/LOLv2/Synthetic/train/")
    val_dataset =retinexDCE_loader_test("Train_data/LOLv2/Synthetic/test/")
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
    
    # train(model, train_loader, val_dataloader, device, save_dir)
    

    test_dataset = retinexDCE_loader_test("Train_data/LOLv2/Synthetic/test/", size=384)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #692->22.85/0.89
    #588->23.7/0.90
    #700->24.4/0.91
    #699->23.65
    save_dir = "./Test_image/LOLv2Syn_prune_ablation_noDenoise"
    # best_weights(model, weights_folder="weights/train_prune/LOLv2Syn_prune_ablation_noDenoise", device=device, test_dataloader=test_dataloader, save_dir=save_dir)
    state_dict = torch.load("./weights/train_prune/LOLv2Syn_prune_ablation_noDenoise/model_epoch_183.pth")
    # model = torch.ao.quantization.quantize_dynamic(
    #     model,  # the original model
    #     {torch.nn.Linear},  # a set of layers to dynamically quantize
    #     dtype=torch.qint8)
    # # # # Create a new state dictionary with the "module." prefix removed from each key
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)  # Load the trained weights
    model.to(device)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(model, test_dataloader, device, save_dir)
    # save_dir = "./Test_image/LOLv2Syn_prune_Dsize_zeroDCE"
    # test_dataset = UnpairedLowLightLoader("Train_data/zeroDCE")
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # test_real(model, test_dataloader, device, save_dir)