from models.modules.ddcm_block import *
from collections import OrderedDict
from torchvision import models
import torch.nn.functional as F
import os

root = '/home/liu/models'
res50_path = os.path.join(root, 'ResNet', 'resnet50-19c8e357.pth')

trained_weights = {
    # 'ddcm_r50': '../weights/Vaihingen_epoch_116.pth',
}


def load_model(name='ddcm_r50', classes=6, load_weights=False, skipp_layer=None):
    if name == 'ddcm_r50':
        model = DDCM_R50(out_channels=classes)
    else:
        print('not found the model')
        return -1

    if load_weights:
        print('-----pretrained weights-----')
        model_dict = model.state_dict()
        mapped_weights = OrderedDict()
        trained_dict = torch.load(trained_weights[name])

        for k_abc in trained_dict.keys():
            print(k_abc)
            if k_abc in model_dict.keys():
                if skipp_layer is None:
                    mapped_weights[k_abc] = trained_dict[k_abc]
                else:
                    if skipp_layer not in k_abc:
                        mapped_weights[k_abc] = trained_dict[k_abc]

        # print(len(mapped_weights), len(model_dict))

        try:
            model.load_state_dict(mapped_weights, strict=False)
        except:
            print("missing some keys in state_dict ... !")
            pass

    return model


def weight_xavier_init(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                # nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class DDCM_R50(nn.Module):
    def __init__(self, out_channels=6, pretrained=True):
        super(DDCM_R50, self).__init__()  # same with  res_fdcs_v5
        self.dec0 = nn.Sequential(
            ddcmBlock(3, 3, [1, 2, 3, 5, 7, 9], kernel=3, bias=False),
            nn.MaxPool2d(kernel_size=2),
        )
        resnet = models.resnet50()
        if pretrained:
            resnet.load_state_dict(torch.load(res50_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3

        self.dec3 = ddcmBlock(1024, 36, [1, 2, 3, 4], kernel=3, bias=False)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.dec1 = ddcmBlock(36, 18, [1], kernel=3, bias=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final = nn.Sequential(
            nn.Conv2d(21, out_channels, kernel_size=3, bias=True, padding=1)
        )

        weight_xavier_init(self.dec0, self.dec1, self.dec3, self.final)

    def forward(self, x):
        x_size = x.size()
        x1 = self.dec0(x)

        x = self.layer3(self.layer2(self.layer1(self.layer0(x))))

        x = self.up4(self.dec3(x))
        x = self.up2(self.dec1(x))
        x = self.final(torch.cat([x, x1], 1))

        return F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

