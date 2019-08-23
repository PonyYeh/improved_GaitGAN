import cv2
import torch as th
import torch.nn as nn


class NetG(nn.Module):
    def __init__(self, nc=3, ngf=96):
        super(NetG, self).__init__()
        self.converter = nn.Sequential(
            nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1, bias=False),# bx1x64x64 --> bx96x32x32
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1,# bx96x32x32 --> bx192x16x16
                      bias=False),
            
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1,# bx192x16x16 --> bx384x8x8
                      bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1,# bx384x8x8 --> bx769x4x4
                      bias=False),
            nn.Dropout(p=0.3, inplace=False), #bug
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(ngf*8, ngf*4,
                               kernel_size=4, stride=2, padding=1, bias=False),# bx769x4x4 --> bx384x8x8
            nn.Dropout(p=0.3, inplace=False), #bug
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True),
#             nn.ReLU(True), #bug

            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, 
                               stride=2, padding=1, bias=False),  # bx384x8x8--> bx192x16x16 
            nn.Dropout(p=0.3, inplace=False), #bug
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),
#             nn.ReLU(True), #bug
            
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4,
                               stride=2, padding=1, bias=False), # bx192x16x16 --> bx96x32x32
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
#             nn.ReLU(True),#bug

            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, # bx96x32x32 --> bx1x64x64 
                               bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.converter(x)
        return x


class NetD(nn.Module):
    def __init__(self, nc=3, ndf=96):
        super(NetD, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x.view(-1, 1)


'''
domain discriminator
'''


class NetA(nn.Module):
    def __init__(self, nc=3, ndf=96):
        super(NetA, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(nc*2, ndf, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x.view(-1, 1)


if __name__ == '__main__':
    netd = NetD()
    a = th.zeros(128, 1, 64, 64)
    b = netd(a)
    print(b.shape)
