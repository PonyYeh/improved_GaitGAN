import cv2
import torch as th
import torch.nn as nn
from spectral import SpectralNorm

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(th.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  th.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = th.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention
    
class NetG(nn.Module):
    def __init__(self, nc=3, ngf=96):
        super(NetG, self).__init__()
        self.encoder = nn.Sequential(
            SpectralNorm(nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1, bias=False)),#bx1x64x64 --> bx96x32x32
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1,#bx96x32x32 --> bx192x16x16
                      bias=False)),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, #bx192x16x16 --> bx384x8x8
                      bias=False)),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, #bx384x8x8 --> bx768x4x4
                      bias=False)),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True),
        )
        self.decoder = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(ngf*8, ngf*4,
                               kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),
#             nn.ReLU(True), #bug

            SpectralNorm(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4,
                               stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),
#             nn.ReLU(True), #bug

            SpectralNorm(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4,
                               stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
#             nn.ReLU(True),#bug
        )
        self.attn = Self_Attn( ngf, 'relu')
        self.last = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        out = self.decoder(encode)
#         print("1",out.shape)
        out,p1 = self.attn(out)
#         print("2",out.shape)
#         print("modelG_Self attn",out.shape,p1.shape)
        out=self.last(out)
#         print("3",out.shape)
        
        return out, encode


class NetD(nn.Module):
    def __init__(self, nc=3, ndf=96):
        super(NetD, self).__init__()
        self.discriminator = nn.Sequential(
            SpectralNorm(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)), #bx1x64x64 --> bx96x32x32
            nn.LeakyReLU(0.2, True),
            
            SpectralNorm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False)), #bx96x32x32 --> bx192x16x16
#             nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False)), #bx192x16x16 --> bx384x8x8
#             nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False)),#bx384x8x8 --> bx768x4x4
#             nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            
        )
        self.attn = Self_Attn( ndf*8,'relu')
        self.last = nn.Sequential(
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4, bias=False),   #bx768x4x4--> bx1x1x1
#             nn.Conv2d(ndf*8, 1, kernel_size=(10,3), stride=(10,3), bias=False),
            
#             nn.Sigmoid()  # for WGAN
            
        )

    def forward(self, x):
        out = self.discriminator(x)
        out, p1 = self.attn(out)
        out = self.last(out)
        return out.squeeze(), p1
#         return out.view(-1, 1), p1


'''
domain discriminator
'''


class NetA(nn.Module):
    def __init__(self, nc=3, ndf=96):
        super(NetA, self).__init__()
        self.discriminator = nn.Sequential(
            SpectralNorm(nn.Conv2d(nc*2, ndf, kernel_size=4, stride=2, padding=1,bias=False)),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False)),
#             nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1,bias=False)),
#             nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1,bias=False)),
#             nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
        )  
        self.attn = Self_Attn( ndf*8,'relu')
        self.last = nn.Sequential(
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4, bias=False),
#             nn.Conv2d(ndf*8, 1, kernel_size=(10,3), stride=(10,3), bias=False),
#             nn.Sigmoid() # for WGAN
        )

    def forward(self, x):
        out = self.discriminator(x)
        out, p1 = self.attn(out)
        out = self.last(out)
        return out.squeeze(), p1
#         return out.view(-1, 1), p1
    


if __name__ == '__main__':
    netd = NetD()
    a = th.zeros(128, 3, 64, 64)
    b = netd(a)
    print(b.shape)
