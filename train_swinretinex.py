import torch
from dataload.retinexSwinDceloader import RetinexSwinDCELoader
from torchvision import transforms
from torch.utils.data import DataLoader
from network.swinDce import RetinexSwinDce
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
import loss1
import torch.nn.functional as F

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)




def train():
    save_dir = "./train_newModel_image/Swinretinex_dce_withColor"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()
    
    model = RetinexSwinDce()
    # model.apply(weights_init)
    dataset = RetinexSwinDCELoader("Train_data/lol_dataset1/our485/")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    L_color = loss1.L_color()
    L_spa = loss1.L_spa()
    L_exp = loss1.L_exp(16,0.6)
    L_TV = loss1.L_TV()
    
    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    num_epochs = 200
    best_train_loss = float('inf')
    criterion = VGGLoss(device)
    criterion1 = CharbonnierLoss()
    criterion2 = ColorLoss()
    
    for epoch in range(num_epochs):
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            
            low_output, A, normal_output, low_output_enhanced = model(low_light_imgs, well_lit_imgs)
            print(low_output.shape)
            low_output_enhanced = F.interpolate(low_output_enhanced, size=(224, 224), mode='bilinear')
            normal_output = F.interpolate(normal_output, size=(224, 224), mode='bilinear')
            low_output = F.interpolate(low_output, size=(224, 224), mode='bilinear')
            #loss for enhancer
            Loss_TV = 200*L_TV(A)
            loss_spa = torch.mean(L_spa(low_output_enhanced, low_light_imgs))
            loss_col = 5*torch.mean(L_color(low_output_enhanced))
            loss_exp = 10*torch.mean(L_exp(low_output_enhanced))

            #loss for R * I
            loss_vgg = criterion(normal_output, well_lit_imgs)
            loss_charon = criterion1(normal_output, well_lit_imgs)
            
            #loss for enhanced
            loss_vgg1 = criterion(low_output, well_lit_imgs)
            loss_charon1 = criterion1(low_output, well_lit_imgs)
            # loss_color = criterion2(low_output, well_lit_imgs)
            

            #add loss
            loss =  Loss_TV + loss_spa + loss_col + loss_exp + 0.5*loss_vgg + 0.5*loss_charon + 0.5*loss_vgg1 + 0.5*loss_charon1
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
                save_image(normal_output, save_path, normalize=True)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}')
        if loss < best_train_loss:
            best_train_loss = loss
            torch.save(model.state_dict(), f"./weights/Swinretinex_dce_withColor.pth")
            

if __name__ == "__main__":
    train()
    
    