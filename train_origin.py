import torch
from dataload.data_lowlight import LowLightDataset
from dataload.retinexDCEloader import retinexDCE_loader
from torchvision import transforms
from torch.utils.data import DataLoader
from network.model_swinIR_GAN import SwinIR, SwinEnhancer
import torch.optim as optim
import torch.nn as nn
import os
from torchvision.utils import save_image
from loss import PerceptualLoss, VGGLoss, CharbonnierLoss, CombinedLoss, RetinexLoss, CombinedLoss1, ColorLoss
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from transformers import AutoModelForSequenceClassification
# from accelerate import Accelerator
# from bitsandbytes import quantize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import loss1



# from TransformerEngine

# print(torch.cuda.device_count())
# print(torch.cuda.is_available())

def weights_init_kaiming(m):
    # if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
    #     if m.bias is not None:
    #         nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def main():
    # print(torch.cuda.is_available())

    save_dir = "./train_newModel_image/SwinEnhancer2262sigmoidlargedata"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    # accelerator = Accelerator()
    dataset = retinexDCE_loader("Train_data/lol_dataset1/our485/")
    #split
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, drop_last=True)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, drop_last=True)

    # dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    # model = SwinIR(in_chans=3, img_size=224, window_size=7,
    #                 img_range=1., depths=[2, 2, 6, 2], embed_dim=48, num_heads=[3, 6, 12, 24],
    #                 mlp_ratio=4, upsampler='', resi_connection='1conv')
    model = SwinEnhancer()
    # model = quantize(model)
    # model = model.to(accelerator.device)
    # model.apply(model._init_weights(model))
    # model.apply(weights_init_kaiming)

    # print(torch.cuda.device_count())
    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # model.apply(lambda m: isinstance(m, nn.Conv2d) and nn.init.kaiming_normal_(m.weight))
    # print(model)

    model.to(device)


    # criterion = nn.MSELoss()
    # criterion1 = PerceptualLoss(layers=["relu2_2"], device=device)
    criterion1 = VGGLoss()
    criterion2 = CharbonnierLoss()
    # criterion2 = ColorLoss()
    # criterion = CombinedLoss(device=device)
    # criterion = RetinexLoss()
    # criterion = CombinedLoss1(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # L_color = loss1.L_color()
    L_spa = loss1.L_spa()
    # L_exp = loss1.L_exp(16,0.6)
    # scaler = GradScaler()
    best_val_loss = float('inf')
    num_epochs = 75
    for epoch in range(num_epochs):
        
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            
            outputs = model(low_light_imgs)
           
            # loss = 0.5 * criterion1(outputs, well_lit_imgs)
            # loss += 0.5 * criterion2(outputs, well_lit_imgs)
            loss_spa = torch.mean(L_spa(outputs, low_light_imgs))
            loss_vgg = criterion1(outputs, well_lit_imgs)
            loss_chon = criterion2(outputs, well_lit_imgs)
            # loss_col = 5*torch.mean(L_color(outputs))
            # loss_exp = 10*torch.mean(L_exp(outputs))
            # loss_c = criterion(outputs, well_lit_imgs)
            # loss = loss_spa + loss_col + loss_exp
            loss = loss_spa + loss_vgg + loss_chon
            # print(f"Loss value: {loss.item()}")
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
                save_image(outputs, save_path, normalize=True)
        #validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (low_light_imgs, well_lit_imgs) in  enumerate(val_dataloader):
                low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
                outputs = model(low_light_imgs)
                # loss = criterion(outputs, well_lit_imgs)
                loss_spa = torch.mean(L_spa(outputs, low_light_imgs))
                loss_vgg = criterion1(outputs, well_lit_imgs)
                # loss_col = 5*torch.mean(L_color(outputs))
                # loss_exp = 10*torch.mean(L_exp(outputs))
                loss_c = criterion2(outputs, well_lit_imgs)
                loss = loss_spa + loss_c + loss_vgg
                # loss = loss_spa + loss_c
                val_loss += loss
        val_loss /= len(val_dataloader)
        # scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./weights/SwinEnhancer2262sigmoidlargedata.pth")

        
def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, (low_light_imgs, _) in enumerate(dataloader):
            low_light_imgs = low_light_imgs.to(device)
            outputs = model(low_light_imgs)
            
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            save_image(outputs, save_path, normalize=True)
            
    print("Testing completed and enhanced images saved!")

if __name__ == "__main__":
    main()


    # test_dataset = LowLightDataset(dataset_path="Train_data/lol_dataset/eval15")
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
    # # model = SwinIR(in_chans=3, img_size=128, window_size=8,
    # #                 img_range=1., depths=[2, 1], embed_dim=96, num_heads=[2, 1],
    # #                 mlp_ratio=4, upsampler='nearest+conv', resi_connection='1conv')
    # model = SwinEnhancer()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # state_dict = torch.load("./weights/SwinEnhancer2262sigmoid.pth")

    # # Create a new state dictionary with the "module." prefix removed from each key
    # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(new_state_dict)  # Load the trained weights
    # model.to(device)
    # save_dir = "./Test_image/SwinEnhancer2262sigmoid"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # test_model(model, test_dataloader, device, save_dir)