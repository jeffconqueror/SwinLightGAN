import torch
from dataload.retinexDCEloader import retinexDCE_loader
from torchvision import transforms
from torch.utils.data import DataLoader
from network.retinex_dce import RetinexDce
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train():
    save_dir = "./train_newModel_image/changed_retinex_dce_withColor"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    dataset = retinexDCE_loader("Train_data/lol_dataset1/our485/")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    L_color = loss1.L_color()
    L_spa = loss1.L_spa()
    L_exp = loss1.L_exp(16,0.6)
    L_TV = loss1.L_TV()

    model = RetinexDce()
    model.apply(weights_init)

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
            loss_color = criterion2(low_output, well_lit_imgs)
            

            #add loss
            loss =  Loss_TV + loss_spa + loss_col + loss_exp + 0.5*loss_vgg + 0.5*loss_charon + 0.5*loss_vgg1 + 0.5*loss_charon1 + loss_color
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
                save_image(low_output, save_path, normalize=True)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}')
        if loss < best_train_loss:
            best_train_loss = loss
            torch.save(model.state_dict(), f"./weights/changed_retinex_dce_withColor.pth")


def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, (low_light_imgs, well_lit_imgs) in enumerate(dataloader):
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            low_output, A, normal_output, low_output_enhanced = model(low_light_imgs, well_lit_imgs)
            
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            save_image(low_output, save_path, normalize=True)

if __name__ == "__main__":
    train()
    
    # test_dataset = retinexDCE_loader("Train_data/lol_dataset/eval15/")
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
    # model = RetinexDce()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # state_dict = torch.load("./weights/changed_retinex_dce.pth")

    # # Create a new state dictionary with the "module." prefix removed from each key
    # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(new_state_dict)  # Load the trained weights
    # model.to(device)
    # save_dir = "./Test_image/changed_retinex_dce"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # test_model(model, test_dataloader, device, save_dir)