import torch
import torch.nn as nn
import torchvision.models as models


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

class VGGLoss(nn.Module):
    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features[:36].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)
        return self.loss(vgg_input, vgg_target)

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon * self.epsilon))
        return loss
    
class CombinedLoss(nn.Module):
    def __init__(self, device, weight=0.5, epsilon=1e-3):
        super(CombinedLoss, self).__init__()
        self.vgg_loss = VGGLoss(device)
        self.charbonnier_loss = CharbonnierLoss(epsilon)
        self.alpha = weight

    def forward(self, pred, target):
        vgg = self.vgg_loss(pred, target)
        charbonnier = self.charbonnier_loss(pred, target)
        combined_loss = self.alpha * vgg + (1 - self.alpha) * charbonnier
        return combined_loss
