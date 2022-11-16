import torch
import torch.nn as nn
from torchvision import models

def deconv(in_cs, out_cs):
    return nn.Sequential(
        nn.ConvTranspose2d(in_cs, out_cs, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
        nn.BatchNorm2d(out_cs),
        nn.ReLU(inplace=True),
    )


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class SeqConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, mid_channels, out_channels, up=True):
        super().__init__()
        # mid_channels = out_channels * 2
        if up:
            self.Up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )
        else:
            self.Up_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )

    def forward(self, x):
        return self.Up_conv(x)


class Trans_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Up_conv = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.Up_conv(x)


class Net_Resnet(nn.Module):
    def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints):
        super(Net_Resnet, self).__init__()
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device
        # self.dropout = nn.Dropout(p=0.5)
        # resnet50
        self.UpConv2 = Trans_Conv(2048, 256)                # size: 16*20--32*40
        self.UpConv3 = Trans_Conv(256, 256)               # size: 32*40--64*80
        self.UpConv4 = Trans_Conv(256, 256)               # size: 32*40--64*80

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)
        if train:
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, img):
        f2 = self.SubResnet(img)
        f3 = self.UpConv2(f2[0])
        f4 = self.UpConv3(f3)
        f4 = self.UpConv4(f4)
        out = self.outConv(f4)
        return out
