import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from thop import profile
       
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se_block = SEBlock(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se_block(out)
        out += identity
        return self.relu(out)

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, padding=1),
            # nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 3, padding=1),
            # nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, the total channels will be 'out_channels // 2 + out_channels // 2'
        self.conv_block = UNetConvBlock(out_channels*2, out_channels)  # Change 'in_channels' to 'out_channels'

    def forward(self, x, bridge):
        up = self.up(x)
        # print("up shape:", up.shape)  # Debugging print
        # print("bridge shape:", bridge.shape)  # Debugging print
        out = torch.cat([up, bridge], dim=1)
        # print("Concatenated shape:", out.shape)  # Debugging print
        return self.conv_block(out)


class RefinementLayer(nn.Module):
    def __init__(self, channels):
        super(RefinementLayer, self).__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            SEBlock(channels),
            ResidualBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            SEBlock(channels),
            ResidualBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.refine(x)
    
class SEBlock(nn.Module): #could try reduction in here
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
     
class DecomposeNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3) -> None:
        super(DecomposeNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        SEBlock(channel),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        SEBlock(channel),
                                        ResidualBlock(channel),  # Adding a residual block
                                        # DilatedConvLayer(channel, channel, dilation=2),  
                                        SEBlock(channel),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        SEBlock(channel),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        SEBlock(channel),
                                        ResidualBlock(channel),  # Adding a residual block
                                        # DilatedConvLayer(channel, channel, dilation=2),  
                                        SEBlock(channel),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')
        
    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        # print(featss.shape)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :])
        L        = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

class DarkRegionAttentionModule(nn.Module):
    def __init__(self, channels):
        super(DarkRegionAttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        # Inverse the attention for dark regions
        attention_map = 1 - attention_map
        return x * attention_map


class DenoiseLayer(nn.Module):
    def __init__(self, channels, num_of_layers=6):
        super(DenoiseLayer, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for i in range(num_of_layers - 2):
            if i % 2 != 0:
                layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
                layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
                layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
                layers.append(ResidualBlock(features))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out
    
class HDR_ToneMappingLayer(nn.Module):
    def __init__(self, input_height=224, input_width=224):
        super(HDR_ToneMappingLayer, self).__init__()
        # Global tone mapping factor
        self.global_tone_mapping_factor = nn.Parameter(torch.ones(1))

        # Local tone mapping components
        self.local_tone_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.local_tone_conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Activation functions and normalization
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm([input_height, input_width])  # Assuming input images are 224

    def forward(self, x):
        # Global tone mapping
        global_tone_mapped = torch.log1p(x * self.global_tone_mapping_factor)

        # Local tone mapping
        local_tone_mapped = self.relu(self.local_tone_conv1(x))
        local_tone_mapped = self.relu(self.local_tone_conv2(local_tone_mapped))

        # Combine global and local tone mappings
        combined_tone_mapped = global_tone_mapped + local_tone_mapped

        # Normalization and final activation
        combined_tone_mapped = self.layernorm(combined_tone_mapped)
        combined_tone_mapped = self.sigmoid(combined_tone_mapped)

        return combined_tone_mapped


class IlluminationEnhancerUNet(nn.Module):
    def __init__(self):
        super(IlluminationEnhancerUNet, self).__init__()
        # Reduced channel sizes for each block
        self.encoder1 = UNetConvBlock(4, 32)  # Start with 32 channels
        self.encoder2 = nn.Sequential(UNetConvBlock(32, 64), ResidualBlock(64), SEBlock(64))
        # self.encoder2 = UNetConvBlock(64, 128)
        # self.sb1 = SEBlock(64)
        self.encoder3 = UNetConvBlock(64, 128)

        # Reduced bottom layer size
        self.bottom = UNetConvBlock(128, 256)

        # Corresponding reductions in the decoder path
        self.up3 = UNetUpBlock(256, 128)
        self.up2 = UNetUpBlock(128, 64)
        self.up1 = UNetUpBlock(64, 32)

        # Final output layer remains the same
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),  # Output channel is 1
            HDR_ToneMappingLayer(),
            nn.LeakyReLU(negative_slope=0.01)
        )
    
    def forward(self, R_low, I_low):
        x = torch.cat((R_low, I_low), dim=1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))

        bottom = self.bottom(nn.MaxPool2d(2)(enc3))

        up3 = self.up3(bottom, enc3)
        up2 = self.up2(up3, enc2)
        # up2 = self.sb1(up2)
       
        up1 = self.up1(up2, enc1)

        output = self.final(up1)
        return output


class RetinexUnet(nn.Module):
    def __init__(self):
        super(RetinexUnet, self).__init__()
        self.decompose = DecomposeNet()
        self.illumination_enhancer = IlluminationEnhancerUNet()
        self.refine = RefinementLayer(channels=1)
        self.dark_attention = DarkRegionAttentionModule(channels=1)
        self.denoise = DenoiseLayer(channels=3)
        


    def forward(self, low, high):
        R_low, I_low = self.decompose(low)
        R_high, I_high = self.decompose(high)
        
        # R_low = self.denoise(R_low)
        
        I_low = self.dark_attention(I_low)
        
        enhanced_I_low = self.illumination_enhancer(R_low, I_low)
        enhanced_I_low = self.refine(enhanced_I_low)
        reconstruct = R_low * torch.concat([enhanced_I_low, enhanced_I_low, enhanced_I_low], dim=1)
        reconstruct = self.denoise(reconstruct)
        return R_low, R_high, enhanced_I_low, I_high, reconstruct
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    model = RetinexUnet()
    input_low = torch.randn(1, 3, 224, 224)
    input_high = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input_low, input_high))

    print('FLOPs: ', flops)
    print('Parameters: ', params)
    for name, module in model.named_children():
        print(f"{name}: {count_parameters(module)} params")