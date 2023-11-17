import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
       
class BasicConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(BasicConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class DilatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(DilatedConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(in_channels, out_channels)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], dim=1)
        return self.conv_block(out)

class RefinementLayer(nn.Module):
    def __init__(self, channels):
        super(RefinementLayer, self).__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.refine(x)
    
class DecomposeNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3) -> None:
        super(DecomposeNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        ResidualBlock(channel),  # Adding a residual block
                                        # DilatedConvLayer(channel, channel, dilation=2),  
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        ResidualBlock(channel),  # Adding a residual block
                                        # DilatedConvLayer(channel, channel, dilation=2),  
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

class DynamicRangeCompression(nn.Module):
    def __init__(self):
        super(DynamicRangeCompression, self).__init__()
        self.compression_factor = nn.Parameter(torch.ones(1))  # Learnable parameter
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Example: logarithmic compression
        # Ensure x is positive and non-zero for logarithm
        x = torch.sigmoid(self.compression_factor * x + self.offset)
        # x = torch.clamp(x, min=1e-6)
        # return torch.log1p(x * self.compression_factor + self.offset)
        return x

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
    def __init__(self, channels):
        super(DenoiseLayer, self).__init__()
        self.denoise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2),  # Larger kernel for denoising
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.denoise(x)
    
class HDR_ToneMappingLayer(nn.Module):
    def __init__(self, input_height=224, input_width=224):
        super(HDR_ToneMappingLayer, self).__init__()
        # Global tone mapping factor
        self.global_tone_mapping_factor = nn.Parameter(torch.ones(1))

        # Local tone mapping components
        self.local_tone_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.local_tone_conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Activation functions and normalization
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm([input_height, input_width])  # Assuming input images are 256x256

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
        self.encoder1 = UNetConvBlock(4, 64)  # Start with 32 channels
        self.encoder2 = UNetConvBlock(64, 128)
        self.encoder3 = UNetConvBlock(128, 256)

        # Reduced bottom layer size
        self.bottom = UNetConvBlock(256, 512)

        # Corresponding reductions in the decoder path
        self.up3 = UNetUpBlock(512, 256)
        self.up2 = UNetUpBlock(256, 128)
        self.up1 = UNetUpBlock(128, 64)

        # Final output layer remains the same
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),  # Output channel is 1
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
        up1 = self.up1(up2, enc1)

        output = self.final(up1)
        return output


class SimpleRetinexDce(nn.Module):
    def __init__(self):
        super(SimpleRetinexDce, self).__init__()
        self.decompose = DecomposeNet()
        self.illumination_enhancer = IlluminationEnhancerUNet()
        self.dynamic_range = DynamicRangeCompression() 
        self.refine = RefinementLayer(channels=1)
        self.dark_attention = DarkRegionAttentionModule(channels=1)
        self.denoise = DenoiseLayer(channels=3)


    def forward(self, low, high):
        R_low, I_low = self.decompose(low)
        R_high, I_high = self.decompose(high)
        
        R_low = self.denoise(R_low)
        
        I_low = self.dark_attention(I_low)
        enhanced_I_low = self.illumination_enhancer(R_low, I_low)
        # enhanced_I_low = self.dynamic_range(enhanced_I_low)
        enhanced_I_low = self.refine(enhanced_I_low)
        return R_low, R_high, enhanced_I_low, I_high
    
    
if __name__ == "__main__":
    model = SimpleRetinexDce()
    low_light_img = torch.rand(1, 3, 256, 256)  # Example low-light image
    well_lit_img = torch.rand(1, 3, 256, 256)   # Example well-lit image
    output = model(low_light_img, well_lit_img)
    print(output)