import torch.nn as nn
from torchvision import models
import torch


import torch
import torch.nn as nn



class AGDenseUNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):
        super(AGDenseUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.AGDense1 = self._AGDense(in_ch, base_ch*2, repeat_time=2)
        self.AGDense2 = self._AGDense(base_ch*2, base_ch * 4, repeat_time=2)
        self.AGDense3 = self._AGDense(base_ch * 4, base_ch * 8, repeat_time=3)
        self.AGDense4 = self._AGDense(base_ch * 8, base_ch * 16, repeat_time=3)
        self.AGDense5 = self._AGDense(base_ch * 16, base_ch * 32, repeat_time=3)

        self.upConv5 = self._UpConv(base_ch * 32, base_ch * 16)
        self.attBlock1 = self._AttBlock(base_ch * 16, base_ch * 16, base_ch * 8)
        self.up5 = self._AGDense(base_ch * 32, base_ch * 16, repeat_time=3)

        self.upConv4 = self._UpConv(base_ch * 16, base_ch * 8)
        self.attBlock2 = self._AttBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up4 = self._AGDense(base_ch * 16, base_ch * 8, repeat_time=3)

        self.upConv3 = self._UpConv(base_ch * 8, base_ch * 4)
        self.attBlock3 = self._AttBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up3 = self._AGDense(base_ch * 8, base_ch * 4, repeat_time=3)

        self.upConv2 = self._UpConv(base_ch * 4, base_ch * 2)
        self.attBlock4 = self._AttBlock(base_ch * 2, base_ch * 2, base_ch * 1)
        self.up2 = self._AGDense(base_ch * 4, base_ch * 2, repeat_time=2)

        #self.Conv_1x1 = nn.Conv2d(16, out_ch, kernel_size=1, stride=1, padding=0)

        self.outc = nn.Conv2d(base_ch * 2, out_ch * 2, kernel_size=3, stride=1, padding=1)



    def forward(self, x):

        x1 = self.AGDense1(x)

        x2 = self.Maxpool(x1)
        x2 = self.AGDense2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.AGDense3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.AGDense4(x4)


        x5 = self.Maxpool(x4)
        x5 = self.AGDense5(x5)

        d5 = self.upConv5(x5)

        x4 = self.attBlock1(d5, x4)

        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up5(d5)

        d4 = self.upConv4(d5)
        x3 = self.attBlock2(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up4(d4)

        d3 = self.upConv3(d4)
        x2 = self.attBlock3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up3(d3)

        d2 = self.upConv2(d3)
        x1 = self.attBlock4(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up2(d2)

        x = self.outc(d2)
        return x


        # up1_cls = self.up1_conv(x5)
        # up2_cls = self.up2_conv(x6)
        # up3_cls = self.up3_conv(x7)
        # up4_cls = self.up4_conv(x8)

        # return {'output': x, 'up1_cls': d5, 'up2_cls': d4, 'up3_cls': d3, 'up4_cls': d2}




    class _AGDense(nn.Module):
        def __init__(self, ch_in, ch_out, repeat_time):
            super(AGDenseUNet._AGDense, self).__init__()
            '''
            self.RRCNN = nn.Sequential(
                I1R2AttUNet._Recblock(ch_out, repeat_time),
                #I1R2AttUNet._Recblock(ch_out, repeat_time)
            )
            #self.dc = I1R2AttUNet._InceptionV1(ch_in, ch_out, repeat_time)
            '''
            self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
            self.dc = AGDenseUNet._Conv(ch_out, ch_out, repeat_time)


        def forward(self, x):
            x = self.Conv_1x1(x)
            x1 = self.dc(x)
            return x + x1

    class _Recblock(nn.Module):
        def __init__(self, ch_out, repeat_time):
            super(AGDenseUNet._Recblock, self).__init__()
            self.repeat_time = repeat_time
            self.ch_out = ch_out
            self.conv = nn.Sequential(
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            for i in range(self.repeat_time):

                if i == 0:
                    x1 = self.conv(x)

                x1 = self.conv(x + x1)
            return x1

    class _UpConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(AGDenseUNet._UpConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class _Conv(nn.Module):
        def __init__(self, in_ch, out_ch, repeat_time):
            super(AGDenseUNet._Conv, self).__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

            self.repeat_time = repeat_time

        def forward(self, x):
            # x1 = self.inc(x)
            x1 = self.conv(x) +x
            x2 = self.conv2(x1) + x1+x
            x3 = self.conv3(x2) + x2 + x1 + x


            return x3





    class _AttBlock(nn.Module):
        def __init__(self, ing_ch, inl_ch, out_ch):
            super(AGDenseUNet._AttBlock, self).__init__()

            self.Wg = nn.Sequential(
                nn.Conv2d(ing_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_ch),)

            self.Wl = nn.Sequential(
                nn.Conv2d(inl_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_ch),)

            self.psi = nn.Sequential(
                nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x1, x2):
            g1 = self.Wg(x1)

            x1 = self.Wl(x2)

            psi = self.relu(g1 + x1)
            psi = self.psi(psi)

            return x2*psi


if __name__ == '__main__':
    from torchsummary import summary

    net = AGDenseUNet(in_ch=5, out_ch=3).cuda()
    summary(net, (5, 256, 256))



