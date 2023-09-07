import torch
from dataload.data_lowlight import LowLightDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from network.model_swinIR_GAN import SwinIR
import torch.optim as optim
import torch.nn as nn
import os
from torchvision.utils import save_image

# print(torch.cuda.device_count())
# print(torch.cuda.is_available())

def main():
    save_dir = "./enhanced_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    
    dataset = LowLightDataset(dataset_path="Train_data/lol_dataset/our485")
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "./weights/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
    
    # Loading model weights
    state_dict = torch.load(model_path)
    

    
    model.load_state_dict(state_dict['params'])

    model.to(device)  # Here, you missed out specifying the 'device' 

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (low_light_imgs, well_lit_imgs) in enumerate(dataloader):
            
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(low_light_imgs)
            
            save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
            save_image(outputs, save_path, normalize=True, range=(-1, 1))
            
            loss = criterion(outputs, well_lit_imgs)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


if __name__ == "__main__":
    # del tensor_name
    # torch.cuda.empty_cache()

    
    main()