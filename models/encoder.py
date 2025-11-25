"""
Backbones supported by torchvison.
"""
import torch
import torch.nn as nn
import torchvision


class Res101Encoder(nn.Module):
    """
    Resnet101 backbone from deeplabv3
    modify the 'downsample' component in layer2 and/or layer3 and/or layer4 as the vanilla Resnet
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights='resnet101'):
        super().__init__()
        # using pretrained model's weights
        if pretrained_weights == 'deeplabv3':
            self.pretrained_weights = torch.load(
                "checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth", map_location='cpu')
        elif pretrained_weights == 'resnet101':
            self.pretrained_weights = torch.load("checkpoints/resnet101-63fe2227.pth",
                                                 map_location='cpu')
        else:
            self.pretrained_weights = pretrained_weights

        _model = torchvision.models.resnet.resnet101(pretrained=False,
                                                     replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()

    def forward(self, x):
        features = dict()
        low_level_features = dict() # 我们可以创建一个专门的字典来存放低级特征
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)
        # features['down1'] = x
        x = self.backbone["maxpool"](x)
        x = self.backbone["layer1"](x)
        # ======================= 核心修改 1 =======================
        # 保存 layer1 的输出作为低级特征，它有丰富的空间细节
        # ResNet-101 layer1 的输出通道数为 256
        low_level_features['layer1'] = x 
        # ==========================================================

        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)
        features['down2'] = self.reduce1(x)  # 这是高级语义特征
        
        x = self.backbone["layer4"](x)
        features['down3'] = self.reduce2(x)  # 这是更高级的语义特征

        # feature map -> avgpool -> fc -> single value
        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)

        # ======================= 核心修改 2 =======================
        # 修改返回值，将低级特征也一并返回
        return (features, low_level_features, t)
        # ==========================================================

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained_weights is not None:
            keys = list(self.pretrained_weights.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(len(keys)):
                if keys[i] in new_keys:
                    new_dic[keys[i]] = self.pretrained_weights[keys[i]]

            self.load_state_dict(new_dic)

