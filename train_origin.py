import torch
from dataload.data_lowlight import LowLightDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from network.model_swinIR_GAN import SwinIR
import torch.optim as optim
import torch.nn as nn
import os
from torchvision.utils import save_image
# from TransformerEngine

# print(torch.cuda.device_count())
# print(torch.cuda.is_available())

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main():
    save_dir = "./enhanced_images1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    
    dataset = LowLightDataset(dataset_path="Train_data/lol_dataset/our485")
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[1], embed_dim=96, num_heads=[1],
                    mlp_ratio=4, upsampler='pixelshuffle', resi_connection='3conv')
    
    model.apply(weights_init_kaiming)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.apply(lambda m: isinstance(m, nn.Conv2d) and nn.init.kaiming_normal_(m.weight))

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001) 

    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (low_light_imgs, well_lit_imgs) in enumerate(dataloader):
            
            low_light_imgs, well_lit_imgs = low_light_imgs.to(device), well_lit_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(low_light_imgs)
            
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
                save_image(outputs, save_path, normalize=True, range=(-1, 1))
            
            loss = criterion(outputs, well_lit_imgs)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), f"./weights/final_epoch_weights.pth")
        
def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, (low_light_imgs, _) in enumerate(dataloader):
            low_light_imgs = low_light_imgs.to(device)
            outputs = model(low_light_imgs)
            
            save_path = os.path.join(save_dir, f"test_batch_{i}.jpg")
            save_image(outputs, save_path, normalize=True, range=(-1, 1))
            
    print("Testing completed and enhanced images saved!")

if __name__ == "__main__":
    # main()


    test_dataset = LowLightDataset(dataset_path="Train_data/lol_dataset/eval15")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
    model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[1], embed_dim=96, num_heads=[1],
                   mlp_ratio=4, upsampler='pixelshuffle', resi_connection='3conv')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("./weights/final_epoch_weights.pth"))  # Load the trained weights
    model.to(device)
    save_dir = "./test_enhanced_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(model, test_dataloader, device, save_dir)