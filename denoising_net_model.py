import torch
import torch.nn as nn


# 层级注意力机制（完成）
class Structure_Diagram(nn.Module):
    def __init__(self):
        super(Structure_Diagram, self).__init__()

        # 第一层（左）
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        # 第二层（左）
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.PReLU()

        # 第三层（左）
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu6 = nn.PReLU()

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu7 = nn.PReLU()

        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        # 第四层（中）
        self.conv9 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu9 = nn.PReLU()

        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()

        self.conv11 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu11 = nn.PReLU()

        # 第三层（右）
        self.subpixel12 = nn.PixelShuffle(2)

        self.conv12 = nn.Conv2d(in_channels=256+1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu13 = nn.PReLU()

        # 第二层（右）
        self.subpixel14 = nn.PixelShuffle(2)

        self.conv15 = nn.Conv2d(in_channels=256+512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu15 = nn.PReLU()

        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        # 第一层（右）
        self.subpixel17 = nn.PixelShuffle(2)

        self.conv18 = nn.Conv2d(in_channels=128+256+512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu18 = nn.PReLU()

        self.conv19 = nn.Conv2d(in_channels=256+1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu19 = nn.PReLU()

        self.conv20 = nn.Conv2d(in_channels=256+1024, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu20 = nn.PReLU()

        # DM1
        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.avg21 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # DM2
        self.conv22 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.avg22 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # DM3
        self.conv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.avg23 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # 双线性插值
        self.upBil24 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.upBil25 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.upBil26 = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, x):

        # 第一层（左）
        output = self.relu1(self.conv1(x))
        temp1 = output
        output = self.relu2(self.conv2(output))
        add1 = output

        # 第二层（右）
        output = self.relu3(self.conv3(output))
        temp2 = self.avg21(self.conv21(temp1))
        output = torch.add(output, temp2)

        output = self.relu4(self.conv4(output))
        temp3 = output

        output = self.relu5(self.conv5(output))
        add2 = output

        # 第三层（左）
        output = self.relu6(self.conv6(output))
        temp4 = self.avg22(self.conv22(temp3))
        output = torch.add(output, temp4)

        output = self.relu7(self.conv7(output))
        temp5 = output

        output = self.relu8(self.conv8(output))
        add3 = output

        # 第四层（中）
        output = self.relu9(self.conv9(output))
        temp6 = self.avg23(self.conv23(temp5))
        output = torch.add(output, temp6)

        output = self.relu10(self.conv10(output))

        output = self.relu11(self.conv11(output))

        upBil1 = self.upBil24(output)

        # 第三层（右）
        output = self.subpixel12(output)
        output = torch.cat([output, add3], 1)

        output = self.relu12(self.conv12(output))

        output = self.relu13(self.conv13(output))

        upBil2 = self.upBil25(output)

        # 第二层（右）
        output = self.subpixel14(output)
        output = torch.cat([output, add2], 1)

        output = self.relu15(self.conv15(output))

        output = self.relu16(self.conv16(output))

        upBil3 = self.upBil26(output)

        # 第一层（右）
        output = self.subpixel17(output)
        output = torch.cat([output, add1, upBil1], 1)

        output = self.relu18(self.conv18(output))
        output = torch.cat([output, upBil2], 1)

        output = self.relu19(self.conv19(output))
        output = torch.cat([output, upBil3], 1)

        output = self.relu20(self.conv20(output))

        return output

# 多特征通道注意模块（完成）
class Multi_Feature_Channel_Attention_Module(nn.Module):
    def __init__(self):
        super(Multi_Feature_Channel_Attention_Module, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu0 = nn.PReLU()
        self.max0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 第一层
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu2 = nn.PReLU()

        self.globalAvg3 = nn.AdaptiveAvgPool2d((1, 1))

        # 第二层
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu5 = nn.PReLU()

        self.globalAvg6 = nn.AdaptiveAvgPool2d((1, 1))

        # 第三层
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu7 = nn.PReLU()

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu8 = nn.PReLU()

        self.globalAvg9 = nn.AdaptiveAvgPool2d((1, 1))

        # 合并层
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.relu10 = nn.PReLU()

        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.relu11 = nn.PReLU()

    def forward(self, x):

        output = self.relu0(self.conv0(x))
        output = self.max0(output)
        temp1 = output
        temp2 = output
        temp3 = output

        # 第一层
        temp1 = self.relu1(self.conv1(temp1))
        temp1 = self.relu2(self.conv2(temp1))
        temp1 = self.globalAvg3(temp1)

        # 第二层
        temp2 = self.relu4(self.conv4(temp2))
        temp2 = self.relu5(self.conv5(temp2))
        temp2 = self.globalAvg6(temp2)

        # 第三层
        temp3 = self.relu7(self.conv7(temp3))
        temp3 = self.relu8(self.conv8(temp3))
        temp3 = self.globalAvg9(temp3)

        # 合并层
        output = torch.cat([temp1, temp2, temp3], 3)

        output = self.relu10(self.conv10(output))

        output = self.relu11(self.conv11(output))

        # 相乘
        output = output * x

        return output

# 局部注意力机制（完成）
class Local_Attention_Module(nn.Module):
    def __init__(self):
        super(Local_Attention_Module, self).__init__()

        # 第一层（左）
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.PReLU()

        # 第二层（左）
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.PReLU()

        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.PReLU()

        # 第三层（中）
        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu7 = nn.PReLU()

        self.conv8 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=2028, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu9 = nn.PReLU()

        # 第二层（右）
        self.subpixel10 = nn.PixelShuffle(2)

        self.conv11 = nn.Conv2d(in_channels=256*3, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu11 = nn.PReLU()

        self.conv12 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        # 第一层（右）
        self.subpixel13 = nn.PixelShuffle(2)

        self.conv14 = nn.Conv2d(in_channels=128 * 3, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()

        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu15 = nn.PReLU()

        self.conv16 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.softMax = nn.Softmax()

    def forward(self, x):

        # 第一层（左）
        output = self.relu1(self.conv1(x))
        output = self.relu2(self.conv2(output))
        output = self.relu3(self.conv3(output))
        add11 = output

        # 降维
        resize1 = torch.reshape(output, (output.size(0), 512, -1))
        # 转置
        resize2 = resize1.transpose(1, 2)

        add12 = self.softMax(torch.matmul(resize1, resize2))
        add12 = torch.matmul(add12, resize1)

        # 升维
        add12 = torch.reshape(output, (add11.size(0), add11.size(1), add11.size(2), add11.size(3)))

        # 第二层（左）
        output = self.max4(output)
        output = self.relu5(self.conv5(output))
        output = self.relu6(self.conv6(output))
        add21 = output

        # 降维
        resize3 = torch.reshape(output, (output.size(0), 1024, -1))
        # 转置
        resize4 = resize3.transpose(1, 2)

        add22 = self.softMax(torch.matmul(resize3, resize4))
        add22 = torch.matmul(add22, resize3)

        # 升维
        add22 = torch.reshape(output, (add21.size(0), add21.size(1), add21.size(2), add21.size(3)))

        # 第三层（中）
        output = self.relu7(self.conv7(output))
        output = self.relu8(self.conv8(output))
        output = self.relu9(self.conv9(output))

        # 第二层（右）
        output = self.subpixel10(output)

        output = torch.cat([add22, add21, output], 1)
        output = self.relu10(self.conv10(output))

        output = self.relu11(self.conv11(output))

        # 第一层（右）
        output = self.subpixel13(output)

        torch.cat([add12, add11, output], 1)
        output = self.relu14(self.conv14(output))

        output = self.relu15(self.conv15(output))

        output = self.relu16(self.conv16(output))

        return output

# 增强学习模块 （完成）
class Reinforcement_Learning_Module(nn.Module):
    def __init__(self):
        super(Reinforcement_Learning_Module, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.relu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.relu5 = nn.PReLU()

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(num_features=256)
        self.relu6 = nn.PReLU()

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(num_features=256)
        self.relu7 = nn.PReLU()

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(num_features=256)
        self.relu8 = nn.PReLU()

    def forward(self, x):

        # 1
        output = self.relu1(self.bn1(self.conv1(x)))
        temp1 = output

        # 2
        output = self.relu2(self.bn2(self.conv2(output)))
        temp2 = output

        # 3
        output = torch.add(output, temp1)
        output = self.relu3(self.bn3(self.conv3(output)))
        temp3 = output

        # 4
        output = torch.add(output, temp2)
        output = self.relu4(self.bn4(self.conv4(output)))
        temp4 = output

        # 5
        output = torch.add(output, temp3)
        output = self.relu5(self.bn5(self.conv5(output)))
        temp5 = output

        # 6
        output = torch.add(output, temp4)
        output = self.relu6(self.bn6(self.conv6(output)))
        temp6 = output

        # 7
        output = torch.add(output, temp5)
        output = self.relu7(self.bn7(self.conv7(output)))

        # 8
        output = torch.add(output, temp6)
        output = self.relu8(self.bn8(self.conv8(output)))

        return output

class Denoising_Net(nn.Module):
    def __init__(self):
        super(Denoising_Net, self).__init__()

        # 第一层（左）
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        # 第二层（左）
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.PReLU()

        # 第三层（中）
        self.max6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv7 = nn.Conv2d(in_channels=256*3, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu7 = nn.PReLU()

        # 第二层（右）
        self.subpixel8 = nn.PixelShuffle(2)

        self.conv9 = nn.Conv2d(in_channels=256+256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu9 = nn.PReLU()

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()

        # 第一层（右）
        self.subpixel11 = nn.PixelShuffle(2)

        self.conv12 = nn.Conv2d(in_channels=128 + 256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu13 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu13 = nn.PReLU()

        self.conv14 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.attention_model_01 = Structure_Diagram()

        self.attention_model_02 = Multi_Feature_Channel_Attention_Module()

        self.attention_model_03 = Local_Attention_Module()

        self.reinforcement_learning_model = Reinforcement_Learning_Module()

    def forward(self, x):

        # 第一层（左）
        output = self.relu1(self.conv_input(x))
        output = self.relu2(self.conv2(output))
        temp1 = output

        # 第二层（左）
        output = self.max3(output)
        output = self.relu4(self.conv4(output))
        output = self.relu5(self.conv5(output))
        temp2 = output

        # 第三层（中）
        output = self.max6(output)

        output1 = self.attention_model_01(output)

        output2 = self.attention_model_02(output)

        output3 = self.attention_model_03(output)

        output4 = self.reinforcement_learning_model(output1)

        output5 = self.reinforcement_learning_model(output2)

        output6 = self.reinforcement_learning_model(output3)

        output = torch.cat([output4, output5, output6], 1)

        output = self.relu7(self.conv7(output))

        # 第二层（右）
        output = self.subpixel8(output)

        output = torch.cat([output, temp2], 1)
        output = self.relu9(self.conv9(output))

        output = self.relu10(self.conv10(output))

        # 第一层（右）
        output = self.subpixel11(output)

        output = torch.cat([output, temp1], 1)
        output = self.relu12(self.conv12(output))

        output = self.relu13(self.conv13(output))

        output = self.relu14(self.conv14(output))

        output = self.conv_output(output)

        return output