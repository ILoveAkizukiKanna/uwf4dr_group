import torch
import torch.nn as nn

# 这个model是在ResNet的Block开始和结尾两个地方加入CBAM

__all__ = ['resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam', 'resnet152_cbam']


def conv1x1(in_channel, out_channel, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channel, out_channel, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        """
        : params: in_planes 输入模块的feature map的channel
        : params: ratio 降维/升维因子
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # fc = shared MLP
        self.fc = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """对空间注意力来说，由于将每个通道中的特征都做同等处理，容易忽略通道间的信息交互"""
        super(SpatialAttention, self).__init__()

        # 这里要保持卷积后的feature尺度不变，必须要padding=kernel_size//2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                               # 输入x = [b, c, 56, 56]
        avg_out = torch.mean(x, dim=1, keepdim=True)    # avg_out = [b, 1, 56, 56]  求x的每个像素在所有channel相同位置上的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # max_out = [b, 1, 56, 56]  求x的每个像素在所有channel相同位置上的最大值
        x = torch.cat([avg_out, max_out], dim=1)        # x = [b, 2, 56, 56]  concat操作
        x = self.sigmoid(self.conv1(x))                 # x = [b, 1, 56, 56]  卷积操作，融合avg和max的信息，全方面考虑
        return x


class CBAM_BasicBlock(nn.Module):
    # resnet18 + resnet34(resdual1)  实线残差结构+虚线残差结构
    expansion = 1  # 残差结构中主分支的卷积核个数是否发生变化（倍数） 第二个卷积核输出是否发生变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        : params: in_channel=第一个conv的输入channel
        : params: out_channel=第一个conv的输出channel
        : params: stride=中间conv的stride
        : params: downsample=None:实线残差结构/Not None:虚线残差结构
        """
        super(CBAM_BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel=in_channel, out_channel=out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channel=out_channel, out_channel=out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class CBAM_Bottleneck(nn.Module):
    # resnet50+resnet101+resnet152（resdual2） 实线残差结构+虚线残差结构
    expansion = 4  # 残差结构中主分支的卷积核个数是否发生变化（倍数）  第三个卷积核输出是否发生变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        : params: in_channel=第一个conv的输入channel
        : params: out_channel=第一个conv的输出channel
        : params: stride=中间conv的stride
                  resnet50/101/152:conv2_x的所有层s=1   conv3_x/conv4_x/conv5_x的第一层s=2,其他层s=1
        : params: downsample=None:实线残差结构/Not None:虚线残差结构
        """
        super(CBAM_Bottleneck, self).__init__()
        # 1x1卷积一般s=1 p=0 => w、h不变   卷积默认向下取整
        self.conv1 = conv1x1(in_channel=in_channel, out_channel=out_channel, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # ----------------------------------------------------------------------------------
        # 3x3卷积一般s=2 p=1 => w、h /2（下采样）     3x3卷积一般s=1 p=1 => w、h不变
        self.conv2 = conv3x3(in_channel=out_channel, out_channel=out_channel, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # ---------------------------------------------------------------------------------
        self.conv3 = conv1x1(in_channel=out_channel, out_channel=out_channel * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # ----------------------------------------------------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class CBAM_ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000):
        """
        : params:  block=BasicBlock/Bottleneck
        : params:  blocks_num=每个layer中残差结构的个数
        : params:  num_classes=数据集的分类个数
        """
        super(CBAM_ResNet, self).__init__()
        self.in_channel = 64  # in_channel=每一个layer层第一个卷积层的输出channel/第一个卷积核的数量

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 池化默认向下取整

        # 在Block之前加入CBAM
        self.ca1 = ChannelAttention(self.in_channel)
        self.sa1 = SpatialAttention()

        # 第1个layer的虚线残差结构只需要改变channel,长、宽不变  所以stride=1
        self.layer1 = self._make_layer(block, blocks_num[0], channel=64, stride=1)
        # 第2/3/4个layer的虚线残差结构不仅要改变channel还要将长、宽缩小为原来的一半 所以stride=2
        self.layer2 = self._make_layer(block, blocks_num[1], channel=128, stride=2)
        self.layer3 = self._make_layer(block, blocks_num[2], channel=256, stride=2)
        self.layer4 = self._make_layer(block, blocks_num[3], channel=512, stride=2)

        # 在Block之后加入CBAM
        self.ca2 = ChannelAttention(2048)  # 最后一个Block后输出[2048, 7, 7]
        self.sa2 = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # AdaptiveAvgPool2d 自适应池化层  output_size=(1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 凯明初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, block_num, channel, stride=1):
        """
        : params: block=BasicBlock/Bottleneck   18/34用BasicBlock   50/101/152用Bottleneck
        : params: block_num=当前layer中残差结构的个数
        : params: channel=每个convx_x中第一个卷积核的数量  每一个layer的这个参数都是固定的
        : params: stride=每个convx_x中第一层中3x3卷积层的stride=每个convx_x中downsample(res)的stride
                  resnet50/101/152   conv2_x=>s=1  conv3_x/conv4_x/conv5_x=>s=2
        """
        downsample = None

        # in_channel:每个convx_x中第一层的第一个卷积核的数量
        # channel*block.expansion:每一个layer最后一个卷积核的数量
        # res50/101/152的conv2/3/4/5_x的in_channel != channel * block.expansion永远成立，所以第一层必有downsample（虚线残差结构）
        # 但是conv2_x的第一层只改变channel不改变w/h（s=1），而conv3_x/conv4_x/conv5_x的第一层不仅改变channel还改变w/h(s=2下采样)
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        # 第一层（含虚线残差结构）加入layers
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))
        # 经过第一层后channel变了
        self.in_channel = channel * block.expansion

        # res50/101/152的conv2/3/4/5_x除了第一层有downsample（虚线残差结构），其他所有层都是实现残差结构（等差映射）
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))  # channel在Bottleneck变化：512->128->512
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # 在Block之前加入CBAM
        out = self.ca1(out) * out
        out = self.sa1(out) * out

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 在Block之后加入CBAM
        out = self.ca2(out) * out
        out = self.sa2(out) * out

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18_cbam(**kwargs):
    """ResNet-18 + CBAM."""
    model = CBAM_ResNet(CBAM_BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_cbam(**kwargs):
    """ResNet-34 + CBAM."""
    model = CBAM_ResNet(CBAM_BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_cbam(**kwargs):
    """ResNet-50 + CBAM."""
    model = CBAM_ResNet(CBAM_Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_cbam(**kwargs):
    """ResNet-101 + CBAM."""
    model = CBAM_ResNet(CBAM_Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_cbam(**kwargs):
    """ResNet-152 + CBAM."""
    model = CBAM_ResNet(CBAM_Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    # 权重测试
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model = resnet50_cbam(num_classes=5)
    print(model)
