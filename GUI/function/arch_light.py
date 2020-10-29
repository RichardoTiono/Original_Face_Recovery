import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



def conv_instance_relu(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )

class Encoder(nn.Module):
    """ Net to general Albedo from features
    """
    def __init__(self):
        super(Encoder, self).__init__()
        #256
        #self.conv1 = nn.Sequential(nn.ReflectionPad2d(3),conv_instance_relu(in_channels = 3, out_channels=64,kernel_size=7,padding=0,stride=1))
        self.conv1 = conv_instance_relu(in_channels = 3, out_channels=64,kernel_size=7,padding=3,stride=1)
        #256
        self.conv2 = conv_instance_relu(in_channels = 64, out_channels=128,kernel_size=4,padding=1,stride=2)
        #128
        self.conv3 = conv_instance_relu(in_channels = 128, out_channels=256,kernel_size=4,padding=1,stride=2)
        #2

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return out3,out2,out1

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Bottleneck(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Bottleneck, self).__init__()
        self.block1 = ResidualBlock(dim_in, dim_out)
        self.block2 = ResidualBlock(dim_in, dim_out)
        self.block3 = ResidualBlock(dim_in, dim_out)
        self.block4 = ResidualBlock(dim_in, dim_out)
        self.block5 = ResidualBlock(dim_in, dim_out)
        self.block6 = ResidualBlock(dim_in, dim_out)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        return out

def deconv_instance_relu(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        #nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )

class Decoder_img(nn.Module):
    def __init__(self):
        super(Decoder_img, self).__init__()
        self.deconv1= deconv_instance_relu(512, 128, kernel_size =4,stride = 2 , padding =1)
        self.deconv2= deconv_instance_relu(256, 64 ,  kernel_size =4,stride = 2 , padding =1)
        #.deconv3= deconv_instance_relu(64, 3 ,  kernel_size =7,stride = 1 , padding =3) 
        self.deconv3 = nn.Sequential(#nn.ReflectionPad2d(1),
                                    nn.Conv2d(128, 3, kernel_size=7, stride=1, padding=3),
                                    nn.Sigmoid())
    
    def forward(self, x, x3, x2, x1):
        x = torch.cat([x,x3],1)  
        out = self.deconv1(x)
        out = torch.cat([out,x2],1)  
        out = self.deconv2(out)        
        out = torch.cat([out,x1],1)  
        out = self.deconv3(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = Encoder()
        self.bottleneck= Bottleneck(256,256)
        self.decoder_img = Decoder_img()

    def forward(self, x):
        out3,out2,out1 = self.encoder(x)
        out = self.bottleneck(out3)
        out_img = self.decoder_img(out,out3,out2,out1)

        return out_img