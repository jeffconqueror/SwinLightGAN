import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import loss1
from network import zeroDCEgen
from dataload import zeroDCEloader
from torchvision.utils import save_image
from tqdm import tqdm
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train():
    save_dir = "./train_newModel_image/zeroDce_with_original"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.cuda.empty_cache()

    dataset = zeroDCEloader.lowlight_loader("Train_data/lol_dataset1/our485/")

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    L_color = loss1.L_color()
    L_spa = loss1.L_spa()
    L_exp = loss1.L_exp(16,0.6)
    L_TV = loss1.L_TV()

    model = zeroDCEgen.ZeroDCEenhancer()
    model.apply(weights_init)

    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    num_epochs = 200
    best_train_loss = float('inf')

    for epoch in range(num_epochs):
        for i, low_light_imgs in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            low_light_imgs = low_light_imgs.to(device)
            optimizer.zero_grad()

            enhanced_img1, output, A = model(low_light_imgs)
            Loss_TV = 200*L_TV(A)
            loss_spa = torch.mean(L_spa(output, low_light_imgs))
            loss_col = 5*torch.mean(L_color(output))
            loss_exp = 10*torch.mean(L_exp(output))
			
            loss =  Loss_TV + loss_spa + loss_col + loss_exp
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.jpg")
                save_image(output, save_path, normalize=True)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}')
        if loss < best_train_loss:
            best_train_loss = loss
            torch.save(model.state_dict(), f"./weights/zeroDce_with_original.pth")

def test_model(model, dataloader, device, save_dir):
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for i, low_light_imgs in enumerate(dataloader):
            low_light_imgs = low_light_imgs.to(device)
            a, outputs, b = model(low_light_imgs)
            
            save_path = os.path.join(save_dir, f"test_batch_{i+1}.jpg")
            save_image(outputs, save_path, normalize=True)
            
    print("Testing completed and enhanced images saved!")


if __name__ == "__main__":
    # train()

    test_dataset = zeroDCEloader.lowlight_loader("Train_data/lol_dataset/eval15/")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
    model = zeroDCEgen.ZeroDCEenhancer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("./weights/zeroDce_with_original.pth")

    # Create a new state dictionary with the "module." prefix removed from each key
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)  # Load the trained weights
    model.to(device)
    save_dir = "./Test_image/zeroDce_with_original"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(model, test_dataloader, device, save_dir)
