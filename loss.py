import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
# import numpy as np
from scipy.ndimage import gaussian_filter
# from pytorch_msssim import ssim

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
    
class PerceptualLoss(nn.Module):
    def __init__(self, layers=["relu2_2"], device="cuda") -> None:
        super(PerceptualLoss, self).__init__()

        self.vgg = models.vgg16(pretrained=True).features.to(device).eval()
        # self.vgg = torch.hub.load('pytorch/vision:v0.13.0', 'vgg16', pretrained=True).features.to(device).eval()
        self.layers = layers
        self.device = device
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def get_features(self, x):
        features = []
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.layers:
                features.append(x)

        return features

    def forward(self, enhanced, ground_truth):
        #normalize the images to match vgg 
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        enhanced = (enhanced - mean) / std
        ground_truth = (ground_truth - mean) / std

        enhanced_features = self.get_features(enhanced)
        ground_truth_features = self.get_features(ground_truth)

        # loss = 0.0
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for ef, gf in zip(enhanced_features, ground_truth_features):
            loss += self.loss(ef, gf)
        
        return loss
        # return torch.tensor(loss, device=self.device, requires_grad=True)

# class VGGLoss(nn.Module):
#     def __init__(self, device="cuda") -> None:
#         super().__init__()
#         self.vgg = models.vgg19(pretrained=True).features[:36].eval().to(device)
#         self.loss = nn.MSELoss()

#         for param in self.vgg.parameters():
#             param.requires_grad = False

#     def forward(self, input, target):
#         vgg_input = self.vgg(input)
#         vgg_target = self.vgg(target)
#         # print("VGG Input Shape:", vgg_input.shape)
#         # print("VGG Target Shape:", vgg_target.shape)
#         return self.loss(vgg_input, vgg_target)

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon * self.epsilon))
        return loss
    
class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.loss = nn.MSELoss()

    def rgb_to_lab(self, img):
        batch_size, _, _, _ = img.shape
        l_channel = []
        a_channel = []
        b_channel = []

        for i in range(batch_size):
            single_img = img[i].detach().permute(1, 2, 0).cpu().numpy()
            # Convert to the range [0, 255]
            single_img = (single_img * 255).astype(np.uint8)
            # Convert to LAB color space using OpenCV
            lab = cv2.cvtColor(single_img, cv2.COLOR_RGB2Lab)
            # Split the LAB image into L, a and b channels
            l, a, b = cv2.split(lab)
            l_channel.append(l)
            a_channel.append(a)
            b_channel.append(b)

        l_channel = np.stack(l_channel, axis=0)
        a_channel = np.stack(a_channel, axis=0)
        b_channel = np.stack(b_channel, axis=0)

        return l_channel, a_channel, b_channel


    def forward(self, pred, target):
        pred_l, pred_a, pred_b = self.rgb_to_lab(pred)
        target_l, target_a, target_b = self.rgb_to_lab(target)

        # Convert numpy arrays back to torch tensors
        pred_a, pred_b = torch.tensor(pred_a).float().to(pred.device), torch.tensor(pred_b).float().to(pred.device)
        target_a, target_b = torch.tensor(target_a).float().to(target.device), torch.tensor(target_b).float().to(target.device)

        # Compute the loss based on the a and b channels
        loss_a = self.loss(pred_a, target_a)
        loss_b = self.loss(pred_b, target_b)

        return loss_a + loss_b
    
class RelativeBrightness(nn.Module):
    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)

    def forward(self, pred, target):
        return torch.abs(torch.mean(pred) - torch.mean(target))

class RelativeStructureLoss(nn.Module):
    def forward(self, pred, target):
        kernel = Variable(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(pred.device))
        pred_gradient_r = torch.abs(F.conv2d(pred[:, 0:1, :, :], kernel))
        pred_gradient_g = torch.abs(F.conv2d(pred[:, 1:2, :, :], kernel))
        pred_gradient_b = torch.abs(F.conv2d(pred[:, 2:3, :, :], kernel))

        # Combine the gradients for each channel (you can average them or use some other method)
        pred_gradient = (pred_gradient_r + pred_gradient_g + pred_gradient_b) / 3.0
        # pred_gradient =  torch.abs(F.conv2d(pred, Variable(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(pred.device))))
        kernel = Variable(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(target.device))
        target_gradient_r = torch.abs(F.conv2d(target[:, 0:1, :, :], kernel))
        target_gradient_g = torch.abs(F.conv2d(target[:, 1:2, :, :], kernel))
        target_gradient_b = torch.abs(F.conv2d(target[:, 2:3, :, :], kernel))

        target_gradient = (target_gradient_r + target_gradient_g + target_gradient_b) / 3.0
        # target_gradient =  torch.abs(F.conv2d(target, Variable(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(target.device))))
        return F.mse_loss(pred_gradient, target_gradient)

class MixedNormLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedNormLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        return self.alpha * l1_loss + (1 - self.alpha) * l2_loss

class CombinedLoss(nn.Module):
    def __init__(self, device, weights=None):
        super(CombinedLoss, self).__init__()
        self.vgg_loss = VGGLoss(device)
        # self.ssim_loss = SSIMLoss()
        self.charbon = CharbonnierLoss()
        self.brightness_loss = RelativeBrightness()
        self.structure_loss = RelativeStructureLoss()
        self.mixed_norm_loss = MixedNormLoss()
        self.weights = weights if weights else {
            'vgg': 0.2,
            # 'ssim': 0.2,
            'charbon': 0.2,
            'brightness': 0.2,
            'structure': 0.2,
            'mixed_norm': 0.2
        }

    def forward(self, pred, target):
        vgg = self.vgg_loss(pred, target)
        # ssim = self.ssim_loss(pred, target)
        char = self.charbon(pred, target)
        brightness = self.brightness_loss(pred, target)
        structure = self.structure_loss(pred, target)
        mixed_norm = self.mixed_norm_loss(pred, target)
        
        combined_loss = (self.weights['vgg'] * vgg + 
                        #  self.weights['ssim'] * ssim + 
                        self.weights['charbon'] * char + 
                         self.weights['brightness'] * brightness + 
                         self.weights['structure'] * structure + 
                         self.weights['mixed_norm'] * mixed_norm)
        return combined_loss

def gaussian_blur(img, sigma):
    # Convert torch tensor to numpy array
    img_np = img.detach().cpu().numpy()
    
    # Apply Gaussian filter
    img_blurred = gaussian_filter(img_np, sigma=sigma)
    
    # Convert back to torch tensor
    return torch.tensor(img_blurred).to(img.device)

class RetinexLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, sigma=1.0):
        super(RetinexLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma  # Standard deviation for the Gaussian filter

    def forward(self, I_enhanced, I_gt):
        # Decompose images into illumination and reflectance
        L_enhanced, R_enhanced = self.decompose(I_enhanced)
        L_gt, R_gt = self.decompose(I_gt)

        # Compute losses
        L_reflectance = F.mse_loss(R_enhanced, R_gt)
        L_illumination = F.mse_loss(L_enhanced, L_gt)

        # Combine losses
        L_total = self.alpha * L_reflectance + self.beta * L_illumination

        return L_total

    def decompose(self, I):
        # Convert image to logarithmic domain
        I_log = torch.log(I + 1e-6)

        # Smooth the image using a Gaussian filter
        # I_smooth = F.gaussian_blur(I_log, kernel_size=5, sigma=(self.sigma, self.sigma))
        I_smooth = gaussian_blur(I_log, sigma=self.sigma)


        # Compute reflectance
        R = I_log - I_smooth

        # Compute illumination
        L = I_log - R

        return torch.exp(L), torch.exp(R)
    


#combine illumination, perceptual, ssim loss
class CombinedLoss1(nn.Module):
    def __init__(self, device):
        super(CombinedLoss1, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, predicted, ground_truth):
        # Illumination Loss
        R_pred, G_pred, B_pred = predicted[:, 0], predicted[:, 1], predicted[:, 2]
        R_gt, G_gt, B_gt = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]
        L_pred = 0.299 * R_pred + 0.587 * G_pred + 0.114 * B_pred
        L_gt = 0.299 * R_gt + 0.587 * G_gt + 0.114 * B_gt
        illumination_loss = F.mse_loss(L_pred, L_gt)

        # Perceptual Loss
        features_pred = self.vgg(predicted)
        features_gt = self.vgg(ground_truth)
        perceptual_loss = F.mse_loss(features_pred, features_gt)

        # SSIM Loss
        ssim_value = ssim(predicted, ground_truth, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_value

        # Combine the losses
        combined_loss = 0.2*illumination_loss + 0.3*perceptual_loss + 0.3*ssim_loss
        return combined_loss
