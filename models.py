from __future__ import annotations
import torch, math, os, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class ResBlock(nn.Module):
    expansion: int = 4
    
    def __init__(self, in_planes: int, out_planes: int | list[int], kernel_size: int, padding: int, stride: Optional[int] = 1) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        self.relu = nn.ReLU(inplace = True)
        self.stride = stride
        self.downsample: bool = False
        
        if in_planes not in out_planes:
            self.downsample = True
    
    
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        residual: torch.tensor = x
        out: torch.tensor = self.conv1(x)
        out: torch.tensor = self.bn1(out)
        out: torch.tensor = self.relu(out)
        
        out: torch.tensor = self.conv2(out)
        out: torch.tensor = self.bn2(out)
        if self.downsample:
            residual: torch.tensor = self.conv1(residual)

        out += residual
        
        out: torch.tensor = self.bn3(out)
        out: torch.tensor = self.relu(out)
        
        return out






class SegmentNetwork(nn.Module):
    def __init__(self) -> None:
        super(SegmentNetwork, self).__init__()
        
        self.res1: ResBlock = ResBlock(1, 8, 7, 3)
        self.res2: ResBlock = ResBlock(8, 8, 3, 1)
        self.res3: ResBlock = ResBlock(8, 16, 3, 1)
        self.res4: ResBlock = ResBlock(16, 32, 3, 1)
        self.res5 ResBlock = ResBlock(32, 64, 3, 1)
        
        self.res_a: ResBlock = ResBlock(64, 64, 1, 0)
        self.res_b: ResBlock = ResBlock(64, 64, 3, 1)
        self.res_c: ResBlock = ResBlock(64, 64, 1, 0)
        self.res_d: ResBlock = ResBlock(64, 64, 3, 1)
        self.res_e: ResBlock = ResBlock(64, 64, 1, 0)
        
        self.res6: ResBlock = ResBlock(64, 3, 1, 0)
        
        self.avg_pool: nn.AvgPool2d = nn.AvgPool2d(2, stride = 2)
        self.deconv: nn.ConvTranspose2d = nn.ConvTranspose2d(3, 3, 16, 16)
    
    
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x: torch.tensor = self.res1(x)
        
        x: torch.tensor = self.res2(x)
        x: torch.tensor = self.avg_pool(x)
        
        x: torch.tensor = self.res3(x)
        x: torch.tensor = self.avg_pool(x)
        
        x: torch.tensor = self.res4(x)
        x: torch.tensor = self.avg_pool(x)
        
        x: torch.tensor = self.res5(x)
        x: torch.tensor = self.avg_pool(x)
        
        x: torch.tensor = self.res_e(self.res_d(self.res_c(self.res_b(self.res_a(x)))))
        
        x: torch.tensor = self.res6(x)
        out: torch.tensor = self.deconv(x)
        return out






class ConvRelu(nn.Module):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU(inplace = True)
    
    
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out: torch.tensor = self.conv(x)
        out: torch.tensor = self.relu(out)
        return out






class DecoderBlock(nn.Module):
    def __init__(self, in_planes: int, middle_planes: int, out_planes: int) -> None:
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            ConvRelu(in_planes, middle_planes),
            nn.ConvTranspose2d(middle_planes, out_planes, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(inplace = True)
        )
    
    
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out: torch.tensor = self.block(x)
        return out







class UNet11(nn.Module):
    def __init__(self, num_filters: Optional[int] = 32, pretrained: Optional[bool] = False, out_filters: Optional[int] = 1) -> None:
        super(UNet11, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained = pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, out_filters, kernel_size=1)



    def forward(self, x: torch.tensor) -> torch.tensor:
        conv1: torch.tensor = self.relu(self.conv1(x))
        conv2: torch.tensor = self.relu(self.conv2(self.pool(conv1)))
        
        conv3s: torch.tensor = self.relu(self.conv3s(self.pool(conv2)))
        conv3: torch.tensor = self.relu(self.conv3(conv3s))
        
        conv4s: torch.tensor = self.relu(self.conv4s(self.pool(conv3)))
        conv4: torch.tensor = self.relu(self.conv4(conv4s))
        
        conv5s: torch.tensor = self.relu(self.conv5s(self.pool(conv4)))
        conv5: torch.tensor = self.relu(self.conv5(conv5s))

        center: torch.tensor = self.center(self.pool(conv5))

        dec5: torch.tensor = self.dec5(torch.cat([center, conv5], 1))
        dec4: torch.tensor = self.dec4(torch.cat([dec5, conv4], 1))
        dec3: torch.tensor = self.dec3(torch.cat([dec4, conv3], 1))
        dec2: torch.tensor = self.dec2(torch.cat([dec3, conv2], 1))
        dec1: torch.tensor = self.dec1(torch.cat([dec2, conv1], 1))

        out: torch.tensor = self.final(dec1)
        return out






class UNet11Classifier(UNet11):
    def __init__(self, num_filters: Optional[int] = 32, pretrained: Optional[bool] = False) -> None:
        super(UNet11Classifier, self).__init__(num_filters = num_filters, pretrained = pretrained)
        self.final_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, 14)
    
    
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x: torch.tensor = super(UNet11Classifier, self).forward(x)
        pooled: torch.tensor = self.final_pool(x)
        out: torch.tensor = self.fc(pooled.view(pooled.size(0), -1))
        return out






class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int, is_deconv: Optional[bool] = True) -> None:
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU(inplace = True)
            )
        
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )



    def forward(self, x: torch.tensor) -> torch.tensor:
        out: torch.tensor = self.block(x)
        return out






class AlbuNet(nn.Module):
    def __init__(self, num_classes: Optional[int] = 1, num_filters: Optional[int] = 32, pretrained: Optional[bool] = False, is_deconv: Optional[bool] = False) -> None:
        super(AlbuNet, self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool
                    )
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)



    def forward(self, x: torch.tensor) -> torch.tensor:
        conv1: torch.tensor= self.conv1(x)
        conv2: torch.tensor = self.conv2(conv1)
        conv3: torch.tensor = self.conv3(conv2)
        conv4: torch.tensor = self.conv4(conv3)
        conv5: torch.tensor = self.conv5(conv4)

        center: torch.tensor = self.center(self.pool(conv5))

        dec5: torch.tensor = self.dec5(torch.cat([center, conv5], 1))

        dec4: torch.tensor = self.dec4(torch.cat([dec5, conv4], 1))
        dec3: torch.tensor = self.dec3(torch.cat([dec4, conv3], 1))
        dec2: torch.tensor = self.dec2(torch.cat([dec3, conv2], 1))
        dec1: torch.tensor = self.dec1(dec2)
        dec0: torch.tensor = self.dec0(dec1)

        if self.num_classes > 1:
            out: torch.tensor = F.log_softmax(self.final(dec0), dim=1)
        else:
            out: torch.tensor = self.final(dec0)

        return out    







class UNet16(nn.Module):
    def __init__(self, num_classes: Optional[int] = 1, num_filters: Optional[int] = 32, pretrained: Optional[bool] = False, is_deconv: Optional[bool] = False, out_filters: Optional[bool] = 1) -> None:
        super(UNet16, self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(self.encoder[10], self.relu, self.encoder[12], self.relu, self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(self.encoder[17], self.relu, self.encoder[19], self.relu, self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(self.encoder[24], self.relu, self.encoder[26], self.relu, self.encoder[28], self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        
        self.final = nn.Conv2d(num_filters, out_filters, kernel_size=1)




    def forward(self, x: torch.tensor) -> torch.tensor:
        conv1: torch.tensor = self.conv1(x)
        conv2: torch.tensor = self.conv2(self.pool(conv1))
        conv3: torch.tensor = self.conv3(self.pool(conv2))
        conv4: torch.tensor = self.conv4(self.pool(conv3))
        conv5: torch.tensor = self.conv5(self.pool(conv4))

        center: torch.tensor = self.center(self.pool(conv5))

        dec5: torch.tensor = self.dec5(torch.cat([center, conv5], 1))
        dec4: torch.tensor = self.dec4(torch.cat([dec5, conv4], 1))
        dec3: torch.tensor = self.dec3(torch.cat([dec4, conv3], 1))
        dec2: torch.tensor = self.dec2(torch.cat([dec3, conv2], 1))
        dec1: torch.tensor = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            out = F.log_softmax(self.final(dec1), dim=1)
        else:
            out = self.final(dec1)

        return out
           
    
    

#@: Driver Code
if __name__.__contains__('__main__'):
    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    
    