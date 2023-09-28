import torch
from dataload.data_lowlight import LowLightDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from network.model_swinIR_GAN import SwinIR
import torch.optim as optim
import torch.nn as nn
import os
from torchvision.utils import save_image
from loss import PerceptualLoss, VGGLoss, CharbonnierLoss, CombinedLoss
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split



# from TransformerEngine

# print(torch.cuda.device_count())
# print(torch.cuda.is_available())

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main():
    # print(torch.cuda.is_available())

    save_dir = "./train_image/enhanced_images_using_combined0.5"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    
    dataset = LowLightDataset(dataset_path="Train_data/lol_dataset/our485")
    #split
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
    # dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    model = SwinIR(in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[1], embed_dim=96, num_heads=[1],
                    mlp_ratio=4, upsampler='', resi_connection='1conv', drop_rate=0.01)
    
    model.apply(weights_init_kaiming)

    # print(torch.cuda.device_count())
    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # model.apply(lambda m: isinstance(m, nn.Conv2d) and nn.init.kaiming_normal_(m.weight))
    # print(model)

    model.to(device)


    # criterion = nn.MSELoss()
    # criterion = PerceptualLoss(layers=["relu2_2"], device=device)
    # criterion = VGGLoss(device)
    # criterion = CharbonnierLoss()
    criterion = CombinedLoss(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005) 
    
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # scaler = GradScaler()
    
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            
            outputs = model(low_light_imgs)
            
            loss = criterion(outputs, well_lit_imgs)
            # print(f"Loss value: {loss.item()}")
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
                save_image(outputs, save_path, normalize=True, range=(-1, 1))
        #validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (low_light_imgs, well_lit_imgs) in  enumerate(val_dataloader):
                low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
                outputs = model(low_light_imgs)
                loss = criterion(outputs, well_lit_imgs)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        # scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')

        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), f"./weights/final_epoch_weights_combine.pth")
        
def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, (low_light_imgs, _) in enumerate(dataloader):
            low_light_imgs = low_light_imgs.to(device)
            outputs = model(low_light_imgs)
            
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            save_image(outputs, save_path, normalize=True, range=(-1, 1))
            
    print("Testing completed and enhanced images saved!")

if __name__ == "__main__":
    main()


    # test_dataset = LowLightDataset(dataset_path="Train_data/lol_dataset/eval15")
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
    # model = SwinIR(in_chans=3, img_size=128, window_size=8,
    #                 img_range=1., depths=[1], embed_dim=96, num_heads=[1],
    #                 mlp_ratio=4, upsampler='', resi_connection='1conv')

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # state_dict = torch.load("./weights/final_epoch_weights_combine.pth")

    # # Create a new state dictionary with the "module." prefix removed from each key
    # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(new_state_dict)  # Load the trained weights
    # model.to(device)
    # save_dir = "./test_enhanced_images"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # test_model(model, test_dataloader, device, save_dir)