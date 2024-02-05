import torch
from dataload.retinexDCEloader import retinexDCE_loader_train, retinexDCE_loader_test
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

# 
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


def train(model, train_loader, val_dataloader, device, save_dir, num_epochs=150):
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
    optimizer = optim.AdamW(model.parameters(), lr=2e-5) 
    # num_epochs = 150
    # best_train_loss = float('inf')
    criterion = VGGLoss()
    criterion1 = CharbonnierLoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for i, (low_light_imgs, well_lit_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
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
            torch.save(model.state_dict(), f"./weights/LOLv2Syn_quan.pth")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./train_quan/LOLv2Syn_quan/training_validation_loss_plot.png')

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
 
def prune(args, model, test_loader, device):

    model = model.to(device)
    summary(model.cuda(), (3,224,224))

    # print(model)
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    # print('factor is : ', bn.numpy())
    # print('factor is : ', bn_avg)
    y, i = torch.sort(bn)
    # print(y)
    thre_index = int(total * args['percent'])
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total

    print('Pre-processing Successful!')
    print('Start to make the real prune')

    acc = test(args, model, test_loader, device, "./train_smallModel/LOLv1_bestAfterquat_test")

    # Make real prune
    print(cfg)
    print(len(cfg_mask))
    newmodel = RetinexUnet()
    if args['cuda']:
        newmodel.cuda()

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args['Prune_save'], "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(acc))

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            print('Input channel index:', idx0)
            # print('weight shape: ', m0.weight.data.shape)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            print("w1 size: ", w1.size())
            print("idx1 max:", idx1.max())
            print("idx1 min:", idx1.min())
            if idx1.max() >= w1.size(0) or idx1.min() < 0:
                raise ValueError("Index out of bounds in idx1")

            w1 = w1[idx0.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            if m0.bias is not None:
                m1.bias.data = m0.bias.data.clone()

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            return x

    delete_layer = []
    last_in_channel = 0
    last_out_channel = newmodel.classifier.in_features
    ii = False
    for k, m in newmodel.named_modules():
        if type(m).__name__ == 'Conv2d':
            if m.in_channels != 0 and m.out_channels == 0:
                last_in_channel = m.in_channels
            if m.in_channels == 0 and m.out_channels != 0:
                replace_layer = k.split('.')
            elif m.in_channels == 0 or m.out_channels == 0:
                delete_layer.append(k.split('.'))
        elif type(m).__name__ == 'BatchNorm2d':
            if m.num_features == 0:
                delete_layer.append(k.split('.'))
                ii = True
            else:
                ii = False
        elif type(m).__name__ == 'ReLU' and ii:
            delete_layer.append(k.split('.'))

    newmodel.get_submodule(replace_layer[0])[int(replace_layer[1])] = nn.Conv2d(last_in_channel, last_out_channel, kernel_size=3, padding=1, bias=False)

    for i, (*parent, k) in enumerate(delete_layer):
        newmodel.get_submodule('.'.join(parent))[int(k)] = Identity()

    print(newmodel)
    model = newmodel.cuda()

    bn_avg = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_avg.append(m.weight.data.abs().mean().item())
    # print('factor is : ', bn.numpy())
    print('Avg bn factor is : ', bn_avg)


    # test(model)
    summary(model.cuda(), (3,32,32))
    return model
    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
    
def ws_quant(model, conv_bits, Linear_bits, device = "CUDA" if torch.cuda.is_available() else "cpu"):

    for k, m in model.named_modules():
        if type(m).__name__ == 'Conv2d':
            # print("Quantizing conv layer : ", k)
            weight = m.weight.data.cpu().numpy()
            num_elements = weight.size
            shape = weight.shape
            # print("weight shape: " ,weight.shape)
            # mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            mat = weight
            num_clusters = 2 ** conv_bits
            if num_elements < num_clusters:
                print(f"Skipping quantization for layer {k} due to small size")
                continue
            min_ = np.min(mat.data)
            max_ = np.max(mat.data)
            space = np.linspace(min_, max_, num=2**conv_bits)
            # kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, algorithm="full")
            # kmeans.fit(mat.data.reshape(-1,1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans.fit(np.reshape(mat.data,(-1,1)))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(shape)
            # print('weight shape: ', new_weight.shape)
            # print('mat shape: ', mat.shape)
            # mat.data = new_weight
            # m.weight.data = torch.from_numpy(mat).to(device)

            scaling_factor = torch.mean(torch.mean(torch.mean(abs(m.weight.data),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            scaling_factor_q = torch.mean(torch.mean(torch.mean(abs(torch.from_numpy(new_weight).to(device)),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            scaling_factor = scaling_factor.to(device) #load to the same GPU
            scaling_factor_diff = (scaling_factor - scaling_factor_q).view(-1,1,1,1)

            # scaling_factor_diff = (scaling_factor / scaling_factor_q).view(-1,1,1,1)

            # print(torch.from_numpy(new_weight).to(device).shape)
            # print(scaling_factor_diff.shape)
            # print('scaling factor difference: ',scaling_factor - scaling_factor_q)

            # m.weight.data = torch.from_numpy(new_weight).to(device)

            m.weight.data = torch.from_numpy(new_weight).to(device) + scaling_factor_diff
            # m.weight.data = torch.mul(torch.from_numpy(new_weight).to(device), scaling_factor_diff)
        
        elif type(m).__name__ == 'Linear':
            # print("Quantizing Linear layer")
            weight = m.weight.data.cpu().numpy()
            shape = weight.shape
            # mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            mat = weight
            num_elements = weight.size
            num_clusters = 2 ** Linear_bits
            if num_elements < num_clusters:
                print(f"Skipping quantization for layer {k} due to small size")
                continue
            min_ = np.min(mat.data)
            max_ = np.max(mat.data)
            space = np.linspace(min_, max_, num=2**Linear_bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, algorithm="full")
            # kmeans.fit(mat.data.reshape(-1,1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans.fit(np.reshape(mat.data,(-1,1)))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(shape)
            # mat.data = new_weight
            # m.weight.data = torch.from_numpy(mat).to(device)
            m.weight.data = torch.from_numpy(new_weight).to(device)
            
def perform_pruning(model, saved_model_path, test_loader, device, pruned_model_path, prune_save_dir, prune_percent=0.6):
    # Load the trained model
    # model = RetinexUnet().to(device)
    state_dict = torch.load(saved_model_path)

    # Create a new state dictionary with the "module." prefix removed from each key
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Define the necessary args
    args = {
        'percent': prune_percent,  # for example, 0.2 for 20% pruning
        'cuda': device.type == 'cuda',
        'Prune_save': prune_save_dir
    }

    # Prune the model
    pruned_model = prune(args, model, test_loader, device)

    # Save the pruned model
    # pruned_model_path = os.path.join(prune_save_dir, 'pruned_model.pth')
    torch.save(pruned_model.state_dict(), pruned_model_path)
    print(f"Pruned model saved at {pruned_model_path}")

    return pruned_model

def fine_tune(model, train_loader, val_dataloader, device, save_dir):
    train(model, train_loader, val_dataloader, device, save_dir)
    
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

if __name__ == "__main__":
    import torch.nn.utils.prune as prune
    model = RetinexUnet()
    print(model.denoise.dncnn[5].se_block.fc[0])
    parameters_to_prune = (
        (model.decompose.net1_convs[0], 'weight'),
        (model.decompose.net1_convs[2], 'weight'),
        (model.decompose.net1_convs[3], 'weight'),
        (model.decompose.net1_convs[5], 'weight'),
        (model.decompose.net1_convs[6], 'weight'),
        (model.decompose.net1_convs[7], 'weight'),
        (model.decompose.net1_convs[8], 'weight'),
        (model.illumination_enhancer.bottom.conv[0], 'weight'),
        (model.illumination_enhancer.bottom.conv[2], 'weight'),
        (model.illumination_enhancer.up2.conv_block.conv[1], 'weight'),
        # (model.refine.refine, 'weight'),
        (model.denoise.dncnn[0], 'weight'),
        (model.denoise.dncnn[2], 'weight'),
        (model.denoise.dncnn[5].se_block.fc[0], 'weight'),
        (model.denoise.dncnn[5].se_block.fc[2], 'weight'),
        (model.denoise.dncnn[6], 'weight'),
        (model.denoise.dncnn[9], 'weight'),
        (model.denoise.dncnn[12].se_block.fc[0], 'weight'),
        (model.denoise.dncnn[12].se_block.fc[2], 'weight'),
        (model.denoise.dncnn[13], 'weight'),
    )
    
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
    


    # prune.random_unstructured(module.net1_convs, name="weight", amount=0.3)
    
    
    
    # save_dir = "./train_quan/LOLv2Syn_quan"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_dataset = retinexDCE_loader_train("Train_data/LOLv2/Synthetic/train/")
    # val_dataset =retinexDCE_loader_test("Train_data/LOLv2/Synthetic/test/")
    # train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
    
    # # train(model, train_loader, val_dataloader, device, save_dir)
    
    # #------------------perform pruning------------------#
    # saved_model_path = "./weights/LOLv2Syn_quan.pth"
    # pruned_model_path = "./weights/LOLv1_bestAfterPrune_test.pth"
    # prune_save_dir = "./pruned_dir"
    # smallest_linear_layer_size = float('inf')

    # smallest_conv_layer_size = float('inf')

    # # for m in model.modules():
    # #     if isinstance(m, nn.Conv2d):
    # #         layer_size = m.weight.data.numel()  # Number of elements in weight tensor
    # #         if layer_size < smallest_conv_layer_size:
    # #             smallest_conv_layer_size = layer_size

    # # print("Smallest convolutional layer size:", smallest_conv_layer_size)

    # # smallest_linear_layer_size = float('inf')

    # # for m in model.modules():
    # #     if isinstance(m, nn.Linear):
    # #         layer_size = m.weight.data.numel()  # Number of elements in weight matrix
    # #         if layer_size < smallest_linear_layer_size:
    # #             smallest_linear_layer_size = layer_size

    # # print("Smallest linear layer size:", smallest_linear_layer_size)

    # # pruned_model = perform_pruning(model, saved_model_path, val_dataloader, device, pruned_model_path, prune_save_dir)
    # ws_quant(model, 8, 8, device)
    # save_dir_quan = "./train_quan/LOLv2Syn_quan"
    # fine_tune(model, train_loader, val_dataloader, device, save_dir_quan)
    # test_dataset = retinexDCE_loader_test("Train_data/LOLv2/Synthetic/test/")
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # state_dict = torch.load("./weights/LOLv2Syn_quan.pth")

    # # Create a new state dictionary with the "module." prefix removed from each key
    # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(new_state_dict)  # Load the trained weights
    # model.to(device)
    # save_dir = "./Test_image/LOLv2Syn_quan"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # test_model(model, test_dataloader, device, save_dir)