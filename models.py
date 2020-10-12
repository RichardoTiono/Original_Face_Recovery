import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import matplotlib.pyplot as plt
from utils import denorm
import numpy as np
from unet_parts import *
import torchvision

def get_shading(N, L):
    c1 = 0.8862269254527579
    c2 = 1.0233267079464883
    c3 = 0.24770795610037571
    c4 = 0.8580855308097834
    c5 = 0.4290427654048917

    nx = N[:, 1, :, :]
    ny = N[:, 0, :, :]
    nz = N[:, 2, :, :]
    b, c, h, w = N.shape
    Y1 = c1 * torch.ones(b, h, w)
    Y2 = c2 * nz
    Y3 = c2 * nx
    Y4 = c2 * ny
    Y5 = c3 * (2 * nz * nz - nx * nx - ny * ny)
    Y6 = c4 * nx * nz
    Y7 = c4 * ny * nz
    Y8 = c5 * (nx * nx - ny * ny)
    Y9 = c4 * nx * ny

    L = L.type(torch.float)
    sh = torch.split(L, 9, dim=1)
    assert(c == len(sh))
    shading = torch.zeros(b, c, h, w)
    
    if torch.cuda.is_available():
        Y1 = Y1.cuda()
        shading = shading.cuda()
    for j in range(c):
        l = sh[j]
        # Scale to 'h x w' dim
        l = l.repeat(1, h*w).view(b, h, w, 9)
        # Convert l into 'batch size', 'Index SH', 'h', 'w'
        l = l.permute([0, 3, 1, 2])
        # Generate shading
        shading[:, j, :, :] = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                            Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                            Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

    return shading


class sfsNetShading(nn.Module):
    def __init__(self):
        super(sfsNetShading, self).__init__()
    
    def forward(self, N, L):
        # Following values are computed from equation
        # from SFSNet
        c1 = 0.8862269254527579
        c2 = 1.0233267079464883
        c3 = 0.24770795610037571
        c4 = 0.8580855308097834
        c5 = 0.4290427654048917

        nx = N[:, 0, :, :]
        ny = N[:, 1, :, :]
        nz = N[:, 2, :, :]
        
        b, c, h, w = N.shape
        
        Y1 = c1 * torch.ones(b, h, w)
        Y2 = c2 * nz
        Y3 = c2 * nx
        Y4 = c2 * ny
        Y5 = c3 * (2 * nz * nz - nx * nx - ny * ny)
        Y6 = c4 * nx * nz
        Y7 = c4 * ny * nz
        Y8 = c5 * (nx * nx - ny * ny)
        Y9 = c4 * nx * ny

        L = L.type(torch.float)
        sh = torch.split(L, 9, dim=1)
        
        assert(c == len(sh))
        shading = torch.zeros(b, c, h, w)
        
        if torch.cuda.is_available():
            Y1 = Y1.cuda()
            shading = shading.cuda()

        for j in range(c):
            l = sh[j]
            # Scale to 'h x w' dim
            l = l.repeat(1, h*w).view(b, h, w, 9)
            # Convert l into 'batch size', 'Index SH', 'h', 'w'
            l = l.permute([0, 3, 1, 2])
            # Generate shading
            shading[:, j, :, :] = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                                Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                                Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

        return shading

def lambertian_attenuation(n):
    # a = [.8862; 1.0233; .4954];
    a = [np.pi * i for i in [1.0, 2 / 3.0, .25]]
    if n > 3:
        sys.stderr.write('don\'t record more than 3 attenuation')
        exit(-1)
    o = a[0:n]
    return o


def normal_harmonics(N, att):
    """
    Return the harmonics evaluated at surface normals N, attenuated by att.
    :param N:
    :param att:
    :return:

    Normals can be scaled surface normals, in which case value of each
    harmonic at each point is scaled by albedo.
    Harmonics written as polynomials
    0,0    1/sqrt(4*pi)
    1,0    z*sqrt(3/(4*pi))
    1,1e    x*sqrt(3/(4*pi))
    1,1o    y*sqrt(3/(4*pi))
    2,0   (2*z.^2 - x.^2 - y.^2)/2 * sqrt(5/(4*pi))
    2,1e  x*z * 3*sqrt(5/(12*pi))
    2,1o  y*z * 3*sqrt(5/(12*pi))
    2,2e  (x.^2-y.^2) * 3*sqrt(5/(48*pi))
    2,2o  x*y * 3*sqrt(5/(12*pi))
    """
    xs = N[0, :].T
    ys = N[1, :].T
    zs = N[2, :].T
    a = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    denom = (a == 0) + a
    # %x = xs./a; y = ys./a; z = zs./a;
    x = xs / denom
    y = ys / denom
    z = zs / denom

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z

    H1 = att[0] * (1 / np.sqrt(4 * np.pi)) * a
    H2 = att[1] * (np.sqrt(3 / (4 * np.pi))) * zs
    H3 = att[1] * (np.sqrt(3 / (4 * np.pi))) * xs
    H4 = att[1] * (np.sqrt(3 / (4 * np.pi))) * ys
    H5 = att[2] * (1 / 2.0) * (np.sqrt(5 / (4 * np.pi))) * ((2 * z2 - x2 - y2) * a)
    H6 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (xz * a)
    H7 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (yz * a)
    H8 = att[2] * (3 * np.sqrt(5 / (48 * np.pi))) * ((x2 - y2) * a)
    H9 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (xy * a)
    H = [H1, H2, H3, H4, H5, H6, H7, H8, H9]

    # --------add by wang -----------
    H = [np.expand_dims(h, axis=1) for h in H]
    H = np.concatenate(H, -1)
    # -------------end---------------
    return H

def create_shading_recon(normals, albedos, lights):
    """
    :type n_out2: np.ndarray
    :type al_out2: np.ndarray
    :type light_out: np.ndarray
    :return:
    """
    #cuman bisa untuk stack 1
    n_out2 = normals.detach().cpu().numpy()
    al_out2 = albedos.detach().cpu().numpy()
    light_out = lights.detach().cpu().numpy()
    n_out2 = np.squeeze(n_out2, 0)
    n_out2 = np.transpose(n_out2, [1, 2, 0])
    n_out2 = 2 * n_out2 - 1  # [-1 1]
    nr = np.sqrt(np.sum(n_out2 ** 2, axis=2))  # nr=sqrt(sum(n_out2.^2,3))
    nr = np.expand_dims(nr, axis=2)
    n_out2 = n_out2 / np.repeat(nr, 3, axis=2)
    al_out2 = np.squeeze(al_out2, 0)
    al_out2 = np.transpose(al_out2, [1, 2, 0])
    light_out = np.transpose(light_out, [1, 0])

    M = n_out2.shape[0]
    No1 = np.reshape(n_out2, (M * M, 3))
    tex1 = np.reshape(al_out2, (M * M, 3))

    la = lambertian_attenuation(3)
    HN1 = normal_harmonics(No1.T, la)

    HS1r = np.matmul(HN1, light_out[0:9])
    HS1g = np.matmul(HN1, light_out[9:18])
    HS1b = np.matmul(HN1, light_out[18:27])

    HS1 = np.zeros(shape=(M, M, 3), dtype=np.float32)
    HS1[:, :, 0] = np.reshape(HS1r, (M, M))
    HS1[:, :, 1] = np.reshape(HS1g, (M, M))
    HS1[:, :, 2] = np.reshape(HS1b, (M, M))
    Tex1 = np.reshape(tex1, (M, M, 3)) * HS1

    IRen0 = Tex1
    Shd = (200 / 255.0) * HS1  # 200 is added instead of 255 so that not to scale the shading to all white
    Ishd0 = Shd
    return [IRen0, Ishd0]

# Base methods for creating convnet
def get_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# SfSNet Models
class ResNetBlock(nn.Module):
    """ Basic building block of ResNet to be used for Normal and Albedo Residual Blocks
    """
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.res = nn.Sequential(
        	nn.BatchNorm2d(in_planes),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(in_planes, in_planes, 3, stride=1, padding=1),
        	nn.BatchNorm2d(in_planes),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(in_planes, out_planes, 3, stride=1, padding=1)
        	)

    def forward(self, x):
        residual = x
        out = self.res(x)
        out += residual

        return out
class baseFeaturesExtractions(nn.Module):
    """ Base Feature extraction
    """
    def __init__(self):
        super(baseFeaturesExtractions, self).__init__()
        self.conv1 = get_conv(3, 64, kernel_size=7, padding=3)
        self.conv2 = get_conv(64, 128, kernel_size=3, padding=1)
        #256
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        #keluar 128
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class NormalResidualBlock(nn.Module):
    """ Net to general Normal from features
    """
    def __init__(self):
        super(NormalResidualBlock, self).__init__()
        self.block1 = ResNetBlock(128, 128)
        self.block2 = ResNetBlock(128, 128)
        self.block3 = ResNetBlock(128, 128)
        self.block4 = ResNetBlock(128, 128)
        self.block5 = ResNetBlock(128, 128)
        self.bn1    = nn.BatchNorm2d(128)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = F.relu(self.bn1(out))
        return out

class AlbedoResidualBlock(nn.Module):
    """ Net to general Albedo from features
    """
    def __init__(self):
        super(AlbedoResidualBlock, self).__init__()
        self.block1 = ResNetBlock(128, 128)
        self.block2 = ResNetBlock(128, 128)
        self.block3 = ResNetBlock(128, 128)
        self.block4 = ResNetBlock(128, 128)
        self.block5 = ResNetBlock(128, 128)
        self.bn1    = nn.BatchNorm2d(128)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = F.relu(self.bn1(out))
        return out

class NormalGenerationNet(nn.Module):
    """ Generating Normal
    """
    def __init__(self):
        super(NormalGenerationNet, self).__init__()
        self.upsample = nn.ConvTranspose2d(128, 128, 4, 2, 1, groups=128, bias=False)
        self.conv1    = get_conv(128, 128, kernel_size=1, stride=1)
        self.conv2    = get_conv(128, 64, kernel_size=3, padding=1)
        self.conv3    = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class AlbedoGenerationNet(nn.Module):
    """ Generating Albedo
    """
    def __init__(self):
        super(AlbedoGenerationNet, self).__init__()
        self.upsample = nn.ConvTranspose2d(128, 128, 4, 2, 1, groups=128, bias=False)
        self.conv1    = get_conv(128, 128, kernel_size=1, stride=1)
        self.conv2    = get_conv(128, 64, kernel_size=3, padding=1)
        self.conv3    = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class LightEstimator(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """
    def __init__(self):
        super(LightEstimator, self).__init__()
        self.conv1 = get_conv(384, 128, kernel_size=1, stride=1)
        self.pool  = nn.AvgPool2d(128, stride=1,padding=0) 
        self.fc    = nn.Linear(128, 27)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        # reshape to batch_size x 128
        out = out.view(-1, 128)
        out = self.fc(out)
        return out



class AlbedoCorrector(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """
    def __init__(self):
        super(AlbedoCorrector, self).__init__()
    #     self.conv1 = get_conv(3, 32, kernel_size=7, stride=1, padding = 3)
    #     self.conv2  = get_conv(32, 64, kernel_size=3, stride=1, padding = 1)
    #     self.conv3  = get_conv(64, 64, kernel_size=3, stride=1, padding = 1)
    #     self.conv4  = get_conv(64, 32, kernel_size=3, stride=1, padding = 1)
    #     self.conv5  = get_conv(32, 3, kernel_size=7, stride=1, padding = 3)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.conv2(out)
    #     out = self.conv3(out)
    #     out = self.conv4(out)
    #     out = self.conv5(out)
    #     out = out+x
    #     return out

        self.encoder = Encoder()
        #self.bottleneck= Bottleneck(256,256)
        self.decoder_img = Decoder_img()

    def forward(self, x):
        out = self.encoder(x)
        #out = self.bottleneck(out)
        out_img = self.decoder_img(out)
        out =  x + out_img
        return out


class ShadingCorrector(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """
    def __init__(self):
        super(ShadingCorrector, self).__init__()
    #     self.conv1 = get_conv(3, 32, kernel_size=7, stride=1, padding = 3)
    #     self.conv2  = get_conv(32, 64, kernel_size=3, stride=1, padding = 1)
    #     self.conv3  = get_conv(64, 64, kernel_size=3, stride=1, padding = 1)
    #     self.conv4  = get_conv(64, 32, kernel_size=3, stride=1, padding = 1)
    #     self.conv5  = get_conv(32, 3, kernel_size=3, stride=1, padding = 1)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.conv2(out)
    #     out = self.conv3(out)
    #     out = self.conv4(out)
    #     out = self.conv5(out)
    #     out = out+x
    #     return out

        self.encoder = Encoder()
        #self.bottleneck= Bottleneck(256,256)
        self.decoder_img = Decoder_img()

    def forward(self, x):
        out = self.encoder(x)
        #out = self.bottleneck(out)
        out_img = self.decoder_img(out)
        out =  x + out_img
        return out

class SfsNetPipeline(nn.Module):
    """ SfSNet Pipeline
    """
    def __init__(self):
        super(SfsNetPipeline, self).__init__()

        self.conv_model            = baseFeaturesExtractions()
        self.normal_residual_model = NormalResidualBlock()
        self.normal_gen_model      = NormalGenerationNet()
        self.albedo_residual_model = AlbedoResidualBlock()
        self.albedo_gen_model      = AlbedoGenerationNet()
        self.light_estimator_model = LightEstimator()
        self.albedo_corrector_model = AlbedoCorrector()
        self.shading_corrector_model = ShadingCorrector()


    def forward(self, face):
        # Following is training pipeline
        # 1. Pass Image from Conv Model to extract features
        out_features = self.conv_model(face)
        # 2 a. Pass Conv features through Normal Residual
        out_normal_features = self.normal_residual_model(out_features)
        # 2 b. Pass Conv features through Albedo Residual
        out_albedo_features = self.albedo_residual_model(out_features)
        # 3 a. Generate Normal
        predicted_normal = self.normal_gen_model(out_normal_features)
        # 3 b. Generate Albedo
        predicted_albedo = self.albedo_gen_model(out_albedo_features)

        all_features = torch.cat((out_features, out_normal_features, out_albedo_features), dim=1)
        # Predict SH
        predicted_sh = self.light_estimator_model(all_features)
        # 4. Generate shading
        out_shading = get_shading(predicted_normal, predicted_sh)
        corrected_albedo = self.albedo_corrector_model(predicted_albedo)
        corrected_shading = self.shading_corrector_model(out_shading)
        out_recon = out_shading * predicted_albedo
        correct_recon = corrected_albedo*corrected_shading
        #out_recon_color = self.color_model(out_recon)


        return predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon , corrected_albedo, corrected_shading,correct_recon

    def fix_weights(self):
        dfs_freeze(self.conv_model)
        dfs_freeze(self.normal_residual_model)
        dfs_freeze(self.normal_gen_model)
        dfs_freeze(self.albedo_residual_model)
        dfs_freeze(self.albedo_gen_model)
        dfs_freeze(self.light_estimator_model)
        # Note that we are not freezing Albedo gen model

        
def conv_instance_relu(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )



def Incept_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding),
        nn.BatchNorm2d(out_channels),
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
        # #64
        # self.conv4 = conv_instance_relu(in_channels = 256, out_channels=512,kernel_size=4,padding=1,stride=2)
        # #32
        # self.conv5 = conv_instance_relu(in_channels = 512, out_channels=512,kernel_size=4,padding=1,stride=2)
        # #16
        # self.conv6 = conv_instance_relu(in_channels = 512, out_channels=512,kernel_size=4,padding=1,stride=2)
        # #8
        # self.conv7 = conv_instance_relu(in_channels = 512, out_channels=512,kernel_size=4,padding=1,stride=2)
        # #4        
        # self.conv8 = conv_instance_relu(in_channels = 512, out_channels=512,kernel_size=4,padding=1,stride=2)
        #2

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        # out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        # out = self.conv8(out)
        return out3,out2,out1

# class ResidualBlock(nn.Module):
#     """Residual Block with instance normalization."""
#     def __init__(self, dim_in, dim_out):
#         super(ResidualBlock, self).__init__()
#         self.main = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

#     def forward(self, x):
#         return x + self.main(x)

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
        nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )

class Decoder_img(nn.Module):
    def __init__(self):
        super(Decoder_img, self).__init__()
        self.deconv1= deconv_instance_relu(512, 128, kernel_size =4,stride = 2 , padding =1)
        self.deconv2= deconv_instance_relu(256, 64 ,  kernel_size =4,stride = 2 , padding =1)
        #self.deconv3= deconv_instance_relu(64, 3 ,  kernel_size =7,stride = 1 , padding =3) 
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

class Decoder_color(nn.Module):
    def __init__(self):
        super(Decoder_color, self).__init__()
        #self.deconv1 = deconv_instance_relu(512, 256, kernel_size =4,stride = 2 , padding =1)
        self.deconv1 = deconv_instance_relu(256, 128, kernel_size =4,stride = 2 , padding =1)
        self.deconv2 = deconv_instance_relu(128, 32 ,  kernel_size =4,stride = 2 , padding =1)
        #self.deconv3 = deconv_instance_relu(64, 3 ,  kernel_size =7,stride = 1 , padding =3) 
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                       nn.Tanh())
    def forward(self, x):
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)  
        #out = self.deconv4(out)  
        return out

class Decoder_cls(nn.Module):
    def __init__(self, image_size=64, conv_in=64, c_dim=4, repeat_num=5):
        super(Decoder_cls, self).__init__()
        layers = []
        conv_dim = conv_in*2
        layers.append(nn.Conv2d(conv_in, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_cls = self.conv2(h)
        return out_cls.view(out_cls.size(0), out_cls.size(1))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = Encoder()
        self.bottleneck= Bottleneck(256,256)
        self.decoder_img = Decoder_img()
        #self.decoder_color = Decoder_color()
        #self.decoder_cls = Decoder_cls()
        #self.constant = nn.Conv2d(3,3,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        out3,out2,out1 = self.encoder(x)
        out = self.bottleneck(out3)
        #out_cls = self.decoder_cls(out)
        # out_cls = self.decoder_cls(torch.narrow(out, 1, 0, 64))
        # #print(out_cls.size(),out_cls)
        # label = out_cls.clone().detach()
        # label -= label.min(1, keepdim=True)[0]
        # label /= label.max(1, keepdim=True)[0]
        # label = torch.floor(label)
        # c = label.view(label.size(0), label.size(1), 1, 1)
        # c = c.repeat(1, 1, out.size(2), out.size(3))
        # color_in = torch.cat([out, c], dim=1)
        out_img = self.decoder_img(out,out3,out2,out1)
        #out_img = self.constant(out_img)

        return out_img
        #return x + out_img

def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers
def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers
    
class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn=='none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, image_size=256, conv_dim=64, c_dim=4, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.01))
        #layers.append(nn.Dropout2d(p=0.5))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)),
            layers.append(nn.BatchNorm2d(curr_dim*2))
            layers.append(nn.LeakyReLU(0.01))
            #layers.append(nn.Dropout2d(p=0.5))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)   
        #self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        #out_cls = self.conv2(h)
        return out_src


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(nn.DataParallel(torchvision.models.vgg16(pretrained=True).features[:4].eval()))
        blocks.append(nn.DataParallel(torchvision.models.vgg16(pretrained=True).features[4:9].eval()))
        blocks.append(nn.DataParallel(torchvision.models.vgg16(pretrained=True).features[9:16].eval()))
        #blocks.append(nn.DataParallel(torchvision.models.vgg16(pretrained=True).features[16:23].eval()))
        for bl in blocks:
            for p in bl.module:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y) #* 0.5**i
        return loss


    # def fix_weights(self):
    #     dfs_freeze(self.main)
    #     dfs_freeze(self.conv1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

      




class Residual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual, self).__init__()
        # nbn1/nbn2/.../nbn5 abn1/abn2/.../abn5
        self.bn = nn.BatchNorm2d(in_channel)
        # nconv1/nconv2/.../nconv5 aconv1/aconv2/.../aconv5
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        # nbn1r/nbn2r/.../nbn5r abn1r/abn2r/.../abn5r
        self.bnr = nn.BatchNorm2d(out_channel)
        # nconv1r/nconv2r/.../nconv5r aconv1r/aconv2r/.../anconv5r
        self.convr = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.convr(F.relu(self.bnr(out)))
        out += x
        return out


class SfSNet(nn.Module):  # SfSNet = PS-Net in SfSNet_deploy.prototxt
    def __init__(self):
        # C64
        super(SfSNet, self).__init__()
        # TODO 初始化器 xavier
        self.conv1 = nn.Conv2d(3, 64, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(64)
        # C128
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        # C128 S2
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1)
        # ------------RESNET for normals------------
        # RES1
        self.n_res1 = Residual(128, 128)
        # RES2
        self.n_res2 = Residual(128, 128)
        # RES3
        self.n_res3 = Residual(128, 128)
        # RES4
        self.n_res4 = Residual(128, 128)
        # RES5
        self.n_res5 = Residual(128, 128)
        # nbn6r
        self.nbn6r = nn.BatchNorm2d(128)
        # CD128
        # TODO 初始化器 bilinear
        self.nup6 = nn.ConvTranspose2d(128, 128, 4, 2, 1, groups=128, bias=False)
        # nconv6
        self.nconv6 = nn.Conv2d(128, 128, 1, 1, 0)
        # nbn6
        self.nbn6 = nn.BatchNorm2d(128)
        # CD 64
        self.nconv7 = nn.Conv2d(128, 64, 3, 1, 1)
        # nbn7
        self.nbn7 = nn.BatchNorm2d(64)
        # C*3
        self.Nconv0 = nn.Conv2d(64, 3, 1, 1, 0)

        # --------------------Albedo---------------
        # RES1
        self.a_res1 = Residual(128, 128)
        # RES2
        self.a_res2 = Residual(128, 128)
        # RES3
        self.a_res3 = Residual(128, 128)
        # RES4
        self.a_res4 = Residual(128, 128)
        # RES5
        self.a_res5 = Residual(128, 128)
        # abn6r
        self.abn6r = nn.BatchNorm2d(128)
        # CD128
        self.aup6 = nn.ConvTranspose2d(128, 128, 4, 2, 1, groups=128, bias=False)
        # nconv6
        self.aconv6 = nn.Conv2d(128, 128, 1, 1, 0)
        # nbn6
        self.abn6 = nn.BatchNorm2d(128)
        # CD 64
        self.aconv7 = nn.Conv2d(128, 64, 3, 1, 1)
        # nbn7
        self.abn7 = nn.BatchNorm2d(64)
        # C*3
        self.Aconv0 = nn.Conv2d(64, 3, 1, 1, 0)

        # ---------------Light------------------
        # lconv1
        self.lconv1 = nn.Conv2d(384, 128, 1, 1, 0)
        # lbn1
        self.lbn1 = nn.BatchNorm2d(128)
        # lpool2r
        self.lpool2r = nn.AvgPool2d(64)
        # fc_light
        self.fc_light = nn.Linear(128, 27)

    def forward(self, inputs):
        permute = [2, 1, 0]
        inputs = inputs[:,permute,:,:]
        # C64
        x = F.relu(self.bn1(self.conv1(inputs)))
        # C128
        x = F.relu(self.bn2(self.conv2(x)))
        # C128 S2
        conv3 = self.conv3(x)
        # ------------RESNET for normals------------
        # RES1
        x = self.n_res1(conv3)
        # RES2
        x = self.n_res2(x)
        # RES3
        x = self.n_res3(x)
        # RES4
        x = self.n_res4(x)
        # RES5
        nsum5 = self.n_res5(x)
        # nbn6r
        nrelu6r = F.relu(self.nbn6r(nsum5))
        # CD128
        x = self.nup6(nrelu6r)
        # nconv6/nbn6/nrelu6
        x = F.relu(self.nbn6(self.nconv6(x)))
        # nconv7/nbn7/nrelu7
        x = F.relu(self.nbn7(self.nconv7(x)))
        # nconv0
        normal = self.Nconv0(x)
        # --------------------Albedo---------------
        # RES1
        x = self.a_res1(conv3)
        # RES2
        x = self.a_res2(x)
        # RES3
        x = self.a_res3(x)
        # RES4
        x = self.a_res4(x)
        # RES5
        asum5 = self.a_res5(x)
        # nbn6r
        arelu6r = F.relu(self.abn6r(asum5))
        # CD128
        x = self.aup6(arelu6r)
        # nconv6/nbn6/nrelu6
        x = F.relu(self.abn6(self.aconv6(x)))
        # nconv7/nbn7/nrelu7
        x = F.relu(self.abn7(self.aconv7(x)))
        # nconv0
        albedo = self.Aconv0(x)
        # ---------------Light------------------
        # lconcat1, shape(1 256 64 64)
        x = torch.cat((nrelu6r, arelu6r), 1)
        # lconcat2, shape(1 384 64 64)
        x = torch.cat([x, conv3], 1)
        # lconv1/lbn1/lrelu1 shape(1 128 64 64)
        x = F.relu(self.lbn1(self.lconv1(x)))
        # lpool2r, shape(1 128 1 1)
        x = self.lpool2r(x)
        x = x.view(-1, 128)
        # fc_light
        light = self.fc_light(x)

        normal = normal[:,permute,:,:]
        albedo = albedo[:,permute,:,:]

        out_shading = get_shading(normal, light)
        out_recon = out_shading *albedo


        # a=out_recon[0].view(3,128,128).permute(1,2,0)
        # a=a.detach().cpu().numpy()
        # plt.imshow(a)
        # plt.show()

        return normal, albedo, light, out_shading ,out_recon

# Use following to fix weights of the model
# Ref - https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/15
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


# Following method loads author provided model weights
# Refer to model_loading_synchronization to getf following mapping
# Following mapping is auto-generated using script
def load_model_from_caffe(src_model, dst_model,src_model2):
    dst_model['conv_model.conv1.0.weight'] = src_model['conv1.weight']
    dst_model['conv_model.conv1.0.bias'] = src_model['conv1.bias']
    dst_model['conv_model.conv1.1.weight'] = src_model['bn1.weight']
    dst_model['conv_model.conv1.1.bias'] = src_model['bn1.bias']
    dst_model['conv_model.conv1.1.running_mean'] = src_model['bn1.running_mean']
    dst_model['conv_model.conv1.1.running_var'] = src_model['bn1.running_var']
    dst_model['conv_model.conv2.0.weight'] = src_model['conv2.weight']
    dst_model['conv_model.conv2.0.bias'] = src_model['conv2.bias']
    dst_model['conv_model.conv2.1.weight'] = src_model['bn2.weight']
    dst_model['conv_model.conv2.1.bias'] = src_model['bn2.bias']
    dst_model['conv_model.conv2.1.running_mean'] = src_model['bn2.running_mean']
    dst_model['conv_model.conv2.1.running_var'] = src_model['bn2.running_var']
    dst_model['conv_model.conv3.weight'] = src_model['conv3.weight']
    dst_model['conv_model.conv3.bias'] = src_model['conv3.bias']
    dst_model['normal_residual_model.block1.res.0.weight'] = src_model['n_res1.bn.weight']
    dst_model['normal_residual_model.block1.res.0.bias'] = src_model['n_res1.bn.bias']
    dst_model['normal_residual_model.block1.res.0.running_mean'] = src_model['n_res1.bn.running_mean']
    dst_model['normal_residual_model.block1.res.0.running_var'] = src_model['n_res1.bn.running_var']
    dst_model['normal_residual_model.block1.res.2.weight'] = src_model['n_res1.conv.weight']
    dst_model['normal_residual_model.block1.res.2.bias'] = src_model['n_res1.conv.bias']
    dst_model['normal_residual_model.block1.res.3.weight'] = src_model['n_res1.bnr.weight']
    dst_model['normal_residual_model.block1.res.3.bias'] = src_model['n_res1.bnr.bias']
    dst_model['normal_residual_model.block1.res.3.running_mean'] = src_model['n_res1.bnr.running_mean']
    dst_model['normal_residual_model.block1.res.3.running_var'] = src_model['n_res1.bnr.running_var']
    dst_model['normal_residual_model.block1.res.5.weight'] = src_model['n_res1.convr.weight']
    dst_model['normal_residual_model.block1.res.5.bias'] = src_model['n_res1.convr.bias']
    dst_model['normal_residual_model.block2.res.0.weight'] = src_model['n_res2.bn.weight']
    dst_model['normal_residual_model.block2.res.0.bias'] = src_model['n_res2.bn.bias']
    dst_model['normal_residual_model.block2.res.0.running_mean'] = src_model['n_res2.bn.running_mean']
    dst_model['normal_residual_model.block2.res.0.running_var'] = src_model['n_res2.bn.running_var']
    dst_model['normal_residual_model.block2.res.2.weight'] = src_model['n_res2.conv.weight']
    dst_model['normal_residual_model.block2.res.2.bias'] = src_model['n_res2.conv.bias']
    dst_model['normal_residual_model.block2.res.3.weight'] = src_model['n_res2.bnr.weight']
    dst_model['normal_residual_model.block2.res.3.bias'] = src_model['n_res2.bnr.bias']
    dst_model['normal_residual_model.block2.res.3.running_mean'] = src_model['n_res2.bnr.running_mean']
    dst_model['normal_residual_model.block2.res.3.running_var'] = src_model['n_res2.bnr.running_var']
    dst_model['normal_residual_model.block2.res.5.weight'] = src_model['n_res2.convr.weight']
    dst_model['normal_residual_model.block2.res.5.bias'] = src_model['n_res2.convr.bias']
    dst_model['normal_residual_model.block3.res.0.weight'] = src_model['n_res3.bn.weight']
    dst_model['normal_residual_model.block3.res.0.bias'] = src_model['n_res3.bn.bias']
    dst_model['normal_residual_model.block3.res.0.running_mean'] = src_model['n_res3.bn.running_mean']
    dst_model['normal_residual_model.block3.res.0.running_var'] = src_model['n_res3.bn.running_var']
    dst_model['normal_residual_model.block3.res.2.weight'] = src_model['n_res3.conv.weight']
    dst_model['normal_residual_model.block3.res.2.bias'] = src_model['n_res3.conv.bias']
    dst_model['normal_residual_model.block3.res.3.weight'] = src_model['n_res3.bnr.weight']
    dst_model['normal_residual_model.block3.res.3.bias'] = src_model['n_res3.bnr.bias']
    dst_model['normal_residual_model.block3.res.3.running_mean'] = src_model['n_res3.bnr.running_mean']
    dst_model['normal_residual_model.block3.res.3.running_var'] = src_model['n_res3.bnr.running_var']
    dst_model['normal_residual_model.block3.res.5.weight'] = src_model['n_res3.convr.weight']
    dst_model['normal_residual_model.block3.res.5.bias'] = src_model['n_res3.convr.bias']
    dst_model['normal_residual_model.block4.res.0.weight'] = src_model['n_res4.bn.weight']
    dst_model['normal_residual_model.block4.res.0.bias'] = src_model['n_res4.bn.bias']
    dst_model['normal_residual_model.block4.res.0.running_mean'] = src_model['n_res4.bn.running_mean']
    dst_model['normal_residual_model.block4.res.0.running_var'] = src_model['n_res4.bn.running_var']
    dst_model['normal_residual_model.block4.res.2.weight'] = src_model['n_res4.conv.weight']
    dst_model['normal_residual_model.block4.res.2.bias'] = src_model['n_res4.conv.bias']
    dst_model['normal_residual_model.block4.res.3.weight'] = src_model['n_res4.bnr.weight']
    dst_model['normal_residual_model.block4.res.3.bias'] = src_model['n_res4.bnr.bias']
    dst_model['normal_residual_model.block4.res.3.running_mean'] = src_model['n_res4.bnr.running_mean']
    dst_model['normal_residual_model.block4.res.3.running_var'] = src_model['n_res4.bnr.running_var']
    dst_model['normal_residual_model.block4.res.5.weight'] = src_model['n_res4.convr.weight']
    dst_model['normal_residual_model.block4.res.5.bias'] = src_model['n_res4.convr.bias']
    dst_model['normal_residual_model.block5.res.0.weight'] = src_model['n_res5.bn.weight']
    dst_model['normal_residual_model.block5.res.0.bias'] = src_model['n_res5.bn.bias']
    dst_model['normal_residual_model.block5.res.0.running_mean'] = src_model['n_res5.bn.running_mean']
    dst_model['normal_residual_model.block5.res.0.running_var'] = src_model['n_res5.bn.running_var']
    dst_model['normal_residual_model.block5.res.2.weight'] = src_model['n_res5.conv.weight']
    dst_model['normal_residual_model.block5.res.2.bias'] = src_model['n_res5.conv.bias']
    dst_model['normal_residual_model.block5.res.3.weight'] = src_model['n_res5.bnr.weight']
    dst_model['normal_residual_model.block5.res.3.bias'] = src_model['n_res5.bnr.bias']
    dst_model['normal_residual_model.block5.res.3.running_mean'] = src_model['n_res5.bnr.running_mean']
    dst_model['normal_residual_model.block5.res.3.running_var'] = src_model['n_res5.bnr.running_var']
    dst_model['normal_residual_model.block5.res.5.weight'] = src_model['n_res5.convr.weight']
    dst_model['normal_residual_model.block5.res.5.bias'] = src_model['n_res5.convr.bias']
    dst_model['normal_residual_model.bn1.weight'] = src_model['nbn6r.weight']
    dst_model['normal_residual_model.bn1.bias'] = src_model['nbn6r.bias']
    dst_model['normal_residual_model.bn1.running_mean'] = src_model['nbn6r.running_mean']
    dst_model['normal_residual_model.bn1.running_var'] = src_model['nbn6r.running_var']
    dst_model['normal_gen_model.upsample.weight'] = src_model['nup6.weight']
    dst_model['normal_gen_model.conv1.0.weight'] = src_model['nconv6.weight']
    dst_model['normal_gen_model.conv1.0.bias'] = src_model['nconv6.bias']
    dst_model['normal_gen_model.conv1.1.weight'] = src_model['nbn6.weight']
    dst_model['normal_gen_model.conv1.1.bias'] = src_model['nbn6.bias']
    dst_model['normal_gen_model.conv1.1.running_mean'] = src_model['nbn6.running_mean']
    dst_model['normal_gen_model.conv1.1.running_var'] = src_model['nbn6.running_var']
    dst_model['normal_gen_model.conv2.0.weight'] = src_model['nconv7.weight']
    dst_model['normal_gen_model.conv2.0.bias'] = src_model['nconv7.bias']
    dst_model['normal_gen_model.conv2.1.weight'] = src_model['nbn7.weight']
    dst_model['normal_gen_model.conv2.1.bias'] = src_model['nbn7.bias']
    dst_model['normal_gen_model.conv2.1.running_mean'] = src_model['nbn7.running_mean']
    dst_model['normal_gen_model.conv2.1.running_var'] = src_model['nbn7.running_var']
    dst_model['normal_gen_model.conv3.weight'] = src_model['Nconv0.weight']
    dst_model['normal_gen_model.conv3.bias'] = src_model['Nconv0.bias']
    dst_model['albedo_residual_model.block1.res.0.weight'] = src_model['a_res1.bn.weight']
    dst_model['albedo_residual_model.block1.res.0.bias'] = src_model['a_res1.bn.bias']
    dst_model['albedo_residual_model.block1.res.0.running_mean'] = src_model['a_res1.bn.running_mean']
    dst_model['albedo_residual_model.block1.res.0.running_var'] = src_model['a_res1.bn.running_var']
    dst_model['albedo_residual_model.block1.res.2.weight'] = src_model['a_res1.conv.weight']
    dst_model['albedo_residual_model.block1.res.2.bias'] = src_model['a_res1.conv.bias']
    dst_model['albedo_residual_model.block1.res.3.weight'] = src_model['a_res1.bnr.weight']
    dst_model['albedo_residual_model.block1.res.3.bias'] = src_model['a_res1.bnr.bias']
    dst_model['albedo_residual_model.block1.res.3.running_mean'] = src_model['a_res1.bnr.running_mean']
    dst_model['albedo_residual_model.block1.res.3.running_var'] = src_model['a_res1.bnr.running_var']
    dst_model['albedo_residual_model.block1.res.5.weight'] = src_model['a_res1.convr.weight']
    dst_model['albedo_residual_model.block1.res.5.bias'] = src_model['a_res1.convr.bias']
    dst_model['albedo_residual_model.block2.res.0.weight'] = src_model['a_res2.bn.weight']
    dst_model['albedo_residual_model.block2.res.0.bias'] = src_model['a_res2.bn.bias']
    dst_model['albedo_residual_model.block2.res.0.running_mean'] = src_model['a_res2.bn.running_mean']
    dst_model['albedo_residual_model.block2.res.0.running_var'] = src_model['a_res2.bn.running_var']
    dst_model['albedo_residual_model.block2.res.2.weight'] = src_model['a_res2.conv.weight']
    dst_model['albedo_residual_model.block2.res.2.bias'] = src_model['a_res2.conv.bias']
    dst_model['albedo_residual_model.block2.res.3.weight'] = src_model['a_res2.bnr.weight']
    dst_model['albedo_residual_model.block2.res.3.bias'] = src_model['a_res2.bnr.bias']
    dst_model['albedo_residual_model.block2.res.3.running_mean'] = src_model['a_res2.bnr.running_mean']
    dst_model['albedo_residual_model.block2.res.3.running_var'] = src_model['a_res2.bnr.running_var']
    dst_model['albedo_residual_model.block2.res.5.weight'] = src_model['a_res2.convr.weight']
    dst_model['albedo_residual_model.block2.res.5.bias'] = src_model['a_res2.convr.bias']
    dst_model['albedo_residual_model.block3.res.0.weight'] = src_model['a_res3.bn.weight']
    dst_model['albedo_residual_model.block3.res.0.bias'] = src_model['a_res3.bn.bias']
    dst_model['albedo_residual_model.block3.res.0.running_mean'] = src_model['a_res3.bn.running_mean']
    dst_model['albedo_residual_model.block3.res.0.running_var'] = src_model['a_res3.bn.running_var']
    dst_model['albedo_residual_model.block3.res.2.weight'] = src_model['a_res3.conv.weight']
    dst_model['albedo_residual_model.block3.res.2.bias'] = src_model['a_res3.conv.bias']
    dst_model['albedo_residual_model.block3.res.3.weight'] = src_model['a_res3.bnr.weight']
    dst_model['albedo_residual_model.block3.res.3.bias'] = src_model['a_res3.bnr.bias']
    dst_model['albedo_residual_model.block3.res.3.running_mean'] = src_model['a_res3.bnr.running_mean']
    dst_model['albedo_residual_model.block3.res.3.running_var'] = src_model['a_res3.bnr.running_var']
    dst_model['albedo_residual_model.block3.res.5.weight'] = src_model['a_res3.convr.weight']
    dst_model['albedo_residual_model.block3.res.5.bias'] = src_model['a_res3.convr.bias']
    dst_model['albedo_residual_model.block4.res.0.weight'] = src_model['a_res4.bn.weight']
    dst_model['albedo_residual_model.block4.res.0.bias'] = src_model['a_res4.bn.bias']
    dst_model['albedo_residual_model.block4.res.0.running_mean'] = src_model['a_res4.bn.running_mean']
    dst_model['albedo_residual_model.block4.res.0.running_var'] = src_model['a_res4.bn.running_var']
    dst_model['albedo_residual_model.block4.res.2.weight'] = src_model['a_res4.conv.weight']
    dst_model['albedo_residual_model.block4.res.2.bias'] = src_model['a_res4.conv.bias']
    dst_model['albedo_residual_model.block4.res.3.weight'] = src_model['a_res4.bnr.weight']
    dst_model['albedo_residual_model.block4.res.3.bias'] = src_model['a_res4.bnr.bias']
    dst_model['albedo_residual_model.block4.res.3.running_mean'] = src_model['a_res4.bnr.running_mean']
    dst_model['albedo_residual_model.block4.res.3.running_var'] = src_model['a_res4.bnr.running_var']
    dst_model['albedo_residual_model.block4.res.5.weight'] = src_model['a_res4.convr.weight']
    dst_model['albedo_residual_model.block4.res.5.bias'] = src_model['a_res4.convr.bias']
    dst_model['albedo_residual_model.block5.res.0.weight'] = src_model['a_res5.bn.weight']
    dst_model['albedo_residual_model.block5.res.0.bias'] = src_model['a_res5.bn.bias']
    dst_model['albedo_residual_model.block5.res.0.running_mean'] = src_model['a_res5.bn.running_mean']
    dst_model['albedo_residual_model.block5.res.0.running_var'] = src_model['a_res5.bn.running_var']
    dst_model['albedo_residual_model.block5.res.2.weight'] = src_model['a_res5.conv.weight']
    dst_model['albedo_residual_model.block5.res.2.bias'] = src_model['a_res5.conv.bias']
    dst_model['albedo_residual_model.block5.res.3.weight'] = src_model['a_res5.bnr.weight']
    dst_model['albedo_residual_model.block5.res.3.bias'] = src_model['a_res5.bnr.bias']
    dst_model['albedo_residual_model.block5.res.3.running_mean'] = src_model['a_res5.bnr.running_mean']
    dst_model['albedo_residual_model.block5.res.3.running_var'] = src_model['a_res5.bnr.running_var']
    dst_model['albedo_residual_model.block5.res.5.weight'] = src_model['a_res5.convr.weight']
    dst_model['albedo_residual_model.block5.res.5.bias'] = src_model['a_res5.convr.bias']
    dst_model['albedo_residual_model.bn1.weight'] = src_model['abn6r.weight']
    dst_model['albedo_residual_model.bn1.bias'] = src_model['abn6r.bias']
    dst_model['albedo_residual_model.bn1.running_mean'] = src_model['abn6r.running_mean']
    dst_model['albedo_residual_model.bn1.running_var'] = src_model['abn6r.running_var']
    dst_model['albedo_gen_model.upsample.weight'] = src_model['aup6.weight']
    dst_model['albedo_gen_model.conv1.0.weight'] = src_model['aconv6.weight']
    dst_model['albedo_gen_model.conv1.0.bias'] = src_model['aconv6.bias']
    dst_model['albedo_gen_model.conv1.1.weight'] = src_model['abn6.weight']
    dst_model['albedo_gen_model.conv1.1.bias'] = src_model['abn6.bias']
    dst_model['albedo_gen_model.conv1.1.running_mean'] = src_model['abn6.running_mean']
    dst_model['albedo_gen_model.conv1.1.running_var'] = src_model['abn6.running_var']
    dst_model['albedo_gen_model.conv2.0.weight'] = src_model['aconv7.weight']
    dst_model['albedo_gen_model.conv2.0.bias'] = src_model['aconv7.bias']
    dst_model['albedo_gen_model.conv2.1.weight'] = src_model['abn7.weight']
    dst_model['albedo_gen_model.conv2.1.bias'] = src_model['abn7.bias']
    dst_model['albedo_gen_model.conv2.1.running_mean'] = src_model['abn7.running_mean']
    dst_model['albedo_gen_model.conv2.1.running_var'] = src_model['abn7.running_var']
    dst_model['albedo_gen_model.conv3.weight'] = src_model['Aconv0.weight']
    dst_model['albedo_gen_model.conv3.bias'] = src_model['Aconv0.bias']
    dst_model['light_estimator_model.conv1.0.weight'] = src_model2['lconv.conv.0.weight']
    dst_model['light_estimator_model.conv1.0.bias'] = src_model2['lconv.conv.0.bias']
    dst_model['light_estimator_model.conv1.1.weight'] = src_model2['lconv.conv.1.weight']
    dst_model['light_estimator_model.conv1.1.bias'] = src_model2['lconv.conv.1.bias']
    dst_model['light_estimator_model.conv1.1.running_mean'] = src_model2['lconv.conv.1.running_mean']
    dst_model['light_estimator_model.conv1.1.running_var'] = src_model2['lconv.conv.1.running_var']
    dst_model['light_estimator_model.fc.weight'] = src_model2['lout.weight']
    dst_model['light_estimator_model.fc.bias'] = src_model2['lout.bias']
    return dst_model

def load_model_from_old(src_model, dst_model):
    dst_model['model.conv_model.conv1.0.weight'] = src_model['model.conv_model.conv1.0.weight']
    dst_model['conv_model.conv1.0.bias'] = src_model['conv1.bias']
    dst_model['conv_model.conv1.1.weight'] = src_model['bn1.weight']
    dst_model['conv_model.conv1.1.bias'] = src_model['bn1.bias']
    dst_model['conv_model.conv1.1.running_mean'] = src_model['bn1.running_mean']
    dst_model['conv_model.conv1.1.running_var'] = src_model['bn1.running_var']
    dst_model['conv_model.conv2.0.weight'] = src_model['conv2.weight']
    dst_model['conv_model.conv2.0.bias'] = src_model['conv2.bias']
    dst_model['conv_model.conv2.1.weight'] = src_model['bn2.weight']
    dst_model['conv_model.conv2.1.bias'] = src_model['bn2.bias']
    dst_model['conv_model.conv2.1.running_mean'] = src_model['bn2.running_mean']
    dst_model['conv_model.conv2.1.running_var'] = src_model['bn2.running_var']
    dst_model['conv_model.conv3.weight'] = src_model['conv3.weight']
    dst_model['conv_model.conv3.bias'] = src_model['conv3.bias']
    dst_model['normal_residual_model.block1.res.0.weight'] = src_model['n_res1.bn.weight']
    dst_model['normal_residual_model.block1.res.0.bias'] = src_model['n_res1.bn.bias']
    dst_model['normal_residual_model.block1.res.0.running_mean'] = src_model['n_res1.bn.running_mean']
    dst_model['normal_residual_model.block1.res.0.running_var'] = src_model['n_res1.bn.running_var']
    dst_model['normal_residual_model.block1.res.2.weight'] = src_model['n_res1.conv.weight']
    dst_model['normal_residual_model.block1.res.2.bias'] = src_model['n_res1.conv.bias']
    dst_model['normal_residual_model.block1.res.3.weight'] = src_model['n_res1.bnr.weight']
    dst_model['normal_residual_model.block1.res.3.bias'] = src_model['n_res1.bnr.bias']
    dst_model['normal_residual_model.block1.res.3.running_mean'] = src_model['n_res1.bnr.running_mean']
    dst_model['normal_residual_model.block1.res.3.running_var'] = src_model['n_res1.bnr.running_var']
    dst_model['normal_residual_model.block1.res.5.weight'] = src_model['n_res1.convr.weight']
    dst_model['normal_residual_model.block1.res.5.bias'] = src_model['n_res1.convr.bias']
    dst_model['normal_residual_model.block2.res.0.weight'] = src_model['n_res2.bn.weight']
    dst_model['normal_residual_model.block2.res.0.bias'] = src_model['n_res2.bn.bias']
    dst_model['normal_residual_model.block2.res.0.running_mean'] = src_model['n_res2.bn.running_mean']
    dst_model['normal_residual_model.block2.res.0.running_var'] = src_model['n_res2.bn.running_var']
    dst_model['normal_residual_model.block2.res.2.weight'] = src_model['n_res2.conv.weight']
    dst_model['normal_residual_model.block2.res.2.bias'] = src_model['n_res2.conv.bias']
    dst_model['normal_residual_model.block2.res.3.weight'] = src_model['n_res2.bnr.weight']
    dst_model['normal_residual_model.block2.res.3.bias'] = src_model['n_res2.bnr.bias']
    dst_model['normal_residual_model.block2.res.3.running_mean'] = src_model['n_res2.bnr.running_mean']
    dst_model['normal_residual_model.block2.res.3.running_var'] = src_model['n_res2.bnr.running_var']
    dst_model['normal_residual_model.block2.res.5.weight'] = src_model['n_res2.convr.weight']
    dst_model['normal_residual_model.block2.res.5.bias'] = src_model['n_res2.convr.bias']
    dst_model['normal_residual_model.block3.res.0.weight'] = src_model['n_res3.bn.weight']
    dst_model['normal_residual_model.block3.res.0.bias'] = src_model['n_res3.bn.bias']
    dst_model['normal_residual_model.block3.res.0.running_mean'] = src_model['n_res3.bn.running_mean']
    dst_model['normal_residual_model.block3.res.0.running_var'] = src_model['n_res3.bn.running_var']
    dst_model['normal_residual_model.block3.res.2.weight'] = src_model['n_res3.conv.weight']
    dst_model['normal_residual_model.block3.res.2.bias'] = src_model['n_res3.conv.bias']
    dst_model['normal_residual_model.block3.res.3.weight'] = src_model['n_res3.bnr.weight']
    dst_model['normal_residual_model.block3.res.3.bias'] = src_model['n_res3.bnr.bias']
    dst_model['normal_residual_model.block3.res.3.running_mean'] = src_model['n_res3.bnr.running_mean']
    dst_model['normal_residual_model.block3.res.3.running_var'] = src_model['n_res3.bnr.running_var']
    dst_model['normal_residual_model.block3.res.5.weight'] = src_model['n_res3.convr.weight']
    dst_model['normal_residual_model.block3.res.5.bias'] = src_model['n_res3.convr.bias']
    dst_model['normal_residual_model.block4.res.0.weight'] = src_model['n_res4.bn.weight']
    dst_model['normal_residual_model.block4.res.0.bias'] = src_model['n_res4.bn.bias']
    dst_model['normal_residual_model.block4.res.0.running_mean'] = src_model['n_res4.bn.running_mean']
    dst_model['normal_residual_model.block4.res.0.running_var'] = src_model['n_res4.bn.running_var']
    dst_model['normal_residual_model.block4.res.2.weight'] = src_model['n_res4.conv.weight']
    dst_model['normal_residual_model.block4.res.2.bias'] = src_model['n_res4.conv.bias']
    dst_model['normal_residual_model.block4.res.3.weight'] = src_model['n_res4.bnr.weight']
    dst_model['normal_residual_model.block4.res.3.bias'] = src_model['n_res4.bnr.bias']
    dst_model['normal_residual_model.block4.res.3.running_mean'] = src_model['n_res4.bnr.running_mean']
    dst_model['normal_residual_model.block4.res.3.running_var'] = src_model['n_res4.bnr.running_var']
    dst_model['normal_residual_model.block4.res.5.weight'] = src_model['n_res4.convr.weight']
    dst_model['normal_residual_model.block4.res.5.bias'] = src_model['n_res4.convr.bias']
    dst_model['normal_residual_model.block5.res.0.weight'] = src_model['n_res5.bn.weight']
    dst_model['normal_residual_model.block5.res.0.bias'] = src_model['n_res5.bn.bias']
    dst_model['normal_residual_model.block5.res.0.running_mean'] = src_model['n_res5.bn.running_mean']
    dst_model['normal_residual_model.block5.res.0.running_var'] = src_model['n_res5.bn.running_var']
    dst_model['normal_residual_model.block5.res.2.weight'] = src_model['n_res5.conv.weight']
    dst_model['normal_residual_model.block5.res.2.bias'] = src_model['n_res5.conv.bias']
    dst_model['normal_residual_model.block5.res.3.weight'] = src_model['n_res5.bnr.weight']
    dst_model['normal_residual_model.block5.res.3.bias'] = src_model['n_res5.bnr.bias']
    dst_model['normal_residual_model.block5.res.3.running_mean'] = src_model['n_res5.bnr.running_mean']
    dst_model['normal_residual_model.block5.res.3.running_var'] = src_model['n_res5.bnr.running_var']
    dst_model['normal_residual_model.block5.res.5.weight'] = src_model['n_res5.convr.weight']
    dst_model['normal_residual_model.block5.res.5.bias'] = src_model['n_res5.convr.bias']
    dst_model['normal_residual_model.bn1.weight'] = src_model['nbn6r.weight']
    dst_model['normal_residual_model.bn1.bias'] = src_model['nbn6r.bias']
    dst_model['normal_residual_model.bn1.running_mean'] = src_model['nbn6r.running_mean']
    dst_model['normal_residual_model.bn1.running_var'] = src_model['nbn6r.running_var']
    dst_model['normal_gen_model.upsample.weight'] = src_model['nup6.weight']
    dst_model['normal_gen_model.conv1.0.weight'] = src_model['nconv6.weight']
    dst_model['normal_gen_model.conv1.0.bias'] = src_model['nconv6.bias']
    dst_model['normal_gen_model.conv1.1.weight'] = src_model['nbn6.weight']
    dst_model['normal_gen_model.conv1.1.bias'] = src_model['nbn6.bias']
    dst_model['normal_gen_model.conv1.1.running_mean'] = src_model['nbn6.running_mean']
    dst_model['normal_gen_model.conv1.1.running_var'] = src_model['nbn6.running_var']
    dst_model['normal_gen_model.conv2.0.weight'] = src_model['nconv7.weight']
    dst_model['normal_gen_model.conv2.0.bias'] = src_model['nconv7.bias']
    dst_model['normal_gen_model.conv2.1.weight'] = src_model['nbn7.weight']
    dst_model['normal_gen_model.conv2.1.bias'] = src_model['nbn7.bias']
    dst_model['normal_gen_model.conv2.1.running_mean'] = src_model['nbn7.running_mean']
    dst_model['normal_gen_model.conv2.1.running_var'] = src_model['nbn7.running_var']
    dst_model['normal_gen_model.conv3.weight'] = src_model['Nconv0.weight']
    dst_model['normal_gen_model.conv3.bias'] = src_model['Nconv0.bias']
    dst_model['albedo_residual_model.block1.res.0.weight'] = src_model['a_res1.bn.weight']
    dst_model['albedo_residual_model.block1.res.0.bias'] = src_model['a_res1.bn.bias']
    dst_model['albedo_residual_model.block1.res.0.running_mean'] = src_model['a_res1.bn.running_mean']
    dst_model['albedo_residual_model.block1.res.0.running_var'] = src_model['a_res1.bn.running_var']
    dst_model['albedo_residual_model.block1.res.2.weight'] = src_model['a_res1.conv.weight']
    dst_model['albedo_residual_model.block1.res.2.bias'] = src_model['a_res1.conv.bias']
    dst_model['albedo_residual_model.block1.res.3.weight'] = src_model['a_res1.bnr.weight']
    dst_model['albedo_residual_model.block1.res.3.bias'] = src_model['a_res1.bnr.bias']
    dst_model['albedo_residual_model.block1.res.3.running_mean'] = src_model['a_res1.bnr.running_mean']
    dst_model['albedo_residual_model.block1.res.3.running_var'] = src_model['a_res1.bnr.running_var']
    dst_model['albedo_residual_model.block1.res.5.weight'] = src_model['a_res1.convr.weight']
    dst_model['albedo_residual_model.block1.res.5.bias'] = src_model['a_res1.convr.bias']
    dst_model['albedo_residual_model.block2.res.0.weight'] = src_model['a_res2.bn.weight']
    dst_model['albedo_residual_model.block2.res.0.bias'] = src_model['a_res2.bn.bias']
    dst_model['albedo_residual_model.block2.res.0.running_mean'] = src_model['a_res2.bn.running_mean']
    dst_model['albedo_residual_model.block2.res.0.running_var'] = src_model['a_res2.bn.running_var']
    dst_model['albedo_residual_model.block2.res.2.weight'] = src_model['a_res2.conv.weight']
    dst_model['albedo_residual_model.block2.res.2.bias'] = src_model['a_res2.conv.bias']
    dst_model['albedo_residual_model.block2.res.3.weight'] = src_model['a_res2.bnr.weight']
    dst_model['albedo_residual_model.block2.res.3.bias'] = src_model['a_res2.bnr.bias']
    dst_model['albedo_residual_model.block2.res.3.running_mean'] = src_model['a_res2.bnr.running_mean']
    dst_model['albedo_residual_model.block2.res.3.running_var'] = src_model['a_res2.bnr.running_var']
    dst_model['albedo_residual_model.block2.res.5.weight'] = src_model['a_res2.convr.weight']
    dst_model['albedo_residual_model.block2.res.5.bias'] = src_model['a_res2.convr.bias']
    dst_model['albedo_residual_model.block3.res.0.weight'] = src_model['a_res3.bn.weight']
    dst_model['albedo_residual_model.block3.res.0.bias'] = src_model['a_res3.bn.bias']
    dst_model['albedo_residual_model.block3.res.0.running_mean'] = src_model['a_res3.bn.running_mean']
    dst_model['albedo_residual_model.block3.res.0.running_var'] = src_model['a_res3.bn.running_var']
    dst_model['albedo_residual_model.block3.res.2.weight'] = src_model['a_res3.conv.weight']
    dst_model['albedo_residual_model.block3.res.2.bias'] = src_model['a_res3.conv.bias']
    dst_model['albedo_residual_model.block3.res.3.weight'] = src_model['a_res3.bnr.weight']
    dst_model['albedo_residual_model.block3.res.3.bias'] = src_model['a_res3.bnr.bias']
    dst_model['albedo_residual_model.block3.res.3.running_mean'] = src_model['a_res3.bnr.running_mean']
    dst_model['albedo_residual_model.block3.res.3.running_var'] = src_model['a_res3.bnr.running_var']
    dst_model['albedo_residual_model.block3.res.5.weight'] = src_model['a_res3.convr.weight']
    dst_model['albedo_residual_model.block3.res.5.bias'] = src_model['a_res3.convr.bias']
    dst_model['albedo_residual_model.block4.res.0.weight'] = src_model['a_res4.bn.weight']
    dst_model['albedo_residual_model.block4.res.0.bias'] = src_model['a_res4.bn.bias']
    dst_model['albedo_residual_model.block4.res.0.running_mean'] = src_model['a_res4.bn.running_mean']
    dst_model['albedo_residual_model.block4.res.0.running_var'] = src_model['a_res4.bn.running_var']
    dst_model['albedo_residual_model.block4.res.2.weight'] = src_model['a_res4.conv.weight']
    dst_model['albedo_residual_model.block4.res.2.bias'] = src_model['a_res4.conv.bias']
    dst_model['albedo_residual_model.block4.res.3.weight'] = src_model['a_res4.bnr.weight']
    dst_model['albedo_residual_model.block4.res.3.bias'] = src_model['a_res4.bnr.bias']
    dst_model['albedo_residual_model.block4.res.3.running_mean'] = src_model['a_res4.bnr.running_mean']
    dst_model['albedo_residual_model.block4.res.3.running_var'] = src_model['a_res4.bnr.running_var']
    dst_model['albedo_residual_model.block4.res.5.weight'] = src_model['a_res4.convr.weight']
    dst_model['albedo_residual_model.block4.res.5.bias'] = src_model['a_res4.convr.bias']
    dst_model['albedo_residual_model.block5.res.0.weight'] = src_model['a_res5.bn.weight']
    dst_model['albedo_residual_model.block5.res.0.bias'] = src_model['a_res5.bn.bias']
    dst_model['albedo_residual_model.block5.res.0.running_mean'] = src_model['a_res5.bn.running_mean']
    dst_model['albedo_residual_model.block5.res.0.running_var'] = src_model['a_res5.bn.running_var']
    dst_model['albedo_residual_model.block5.res.2.weight'] = src_model['a_res5.conv.weight']
    dst_model['albedo_residual_model.block5.res.2.bias'] = src_model['a_res5.conv.bias']
    dst_model['albedo_residual_model.block5.res.3.weight'] = src_model['a_res5.bnr.weight']
    dst_model['albedo_residual_model.block5.res.3.bias'] = src_model['a_res5.bnr.bias']
    dst_model['albedo_residual_model.block5.res.3.running_mean'] = src_model['a_res5.bnr.running_mean']
    dst_model['albedo_residual_model.block5.res.3.running_var'] = src_model['a_res5.bnr.running_var']
    dst_model['albedo_residual_model.block5.res.5.weight'] = src_model['a_res5.convr.weight']
    dst_model['albedo_residual_model.block5.res.5.bias'] = src_model['a_res5.convr.bias']
    dst_model['albedo_residual_model.bn1.weight'] = src_model['abn6r.weight']
    dst_model['albedo_residual_model.bn1.bias'] = src_model['abn6r.bias']
    dst_model['albedo_residual_model.bn1.running_mean'] = src_model['abn6r.running_mean']
    dst_model['albedo_residual_model.bn1.running_var'] = src_model['abn6r.running_var']
    dst_model['albedo_gen_model.upsample.weight'] = src_model['aup6.weight']
    dst_model['albedo_gen_model.conv1.0.weight'] = src_model['aconv6.weight']
    dst_model['albedo_gen_model.conv1.0.bias'] = src_model['aconv6.bias']
    dst_model['albedo_gen_model.conv1.1.weight'] = src_model['abn6.weight']
    dst_model['albedo_gen_model.conv1.1.bias'] = src_model['abn6.bias']
    dst_model['albedo_gen_model.conv1.1.running_mean'] = src_model['abn6.running_mean']
    dst_model['albedo_gen_model.conv1.1.running_var'] = src_model['abn6.running_var']
    dst_model['albedo_gen_model.conv2.0.weight'] = src_model['aconv7.weight']
    dst_model['albedo_gen_model.conv2.0.bias'] = src_model['aconv7.bias']
    dst_model['albedo_gen_model.conv2.1.weight'] = src_model['abn7.weight']
    dst_model['albedo_gen_model.conv2.1.bias'] = src_model['abn7.bias']
    dst_model['albedo_gen_model.conv2.1.running_mean'] = src_model['abn7.running_mean']
    dst_model['albedo_gen_model.conv2.1.running_var'] = src_model['abn7.running_var']
    dst_model['albedo_gen_model.conv3.weight'] = src_model['Aconv0.weight']
    dst_model['albedo_gen_model.conv3.bias'] = src_model['Aconv0.bias']
    dst_model['light_estimator_model.conv1.0.weight'] = src_model2['lconv.conv.0.weight']
    dst_model['light_estimator_model.conv1.0.bias'] = src_model2['lconv.conv.0.bias']
    dst_model['light_estimator_model.conv1.1.weight'] = src_model2['lconv.conv.1.weight']
    dst_model['light_estimator_model.conv1.1.bias'] = src_model2['lconv.conv.1.bias']
    dst_model['light_estimator_model.conv1.1.running_mean'] = src_model2['lconv.conv.1.running_mean']
    dst_model['light_estimator_model.conv1.1.running_var'] = src_model2['lconv.conv.1.running_var']
    dst_model['light_estimator_model.fc.weight'] = src_model2['lout.weight']
    dst_model['light_estimator_model.fc.bias'] = src_model2['lout.bias']
    return dst_model

def load_weights_from_pkl(weights_pkl):
        from torch import from_numpy
        with open(weights_pkl, 'rb') as wp:
            try:
                # for python3
                name_weights = pkl.load(wp, encoding='latin1')
            except TypeError as e:
                # for python2
                name_weights = pkl.load(wp)
            state_dict = {}

            def _set_deconv(layer, key):
                state_dict[layer+'.weight'] = from_numpy(name_weights[key]['weight'])

            def _set(layer, key):
                state_dict[layer + '.weight'] = from_numpy(name_weights[key]['weight'])
                state_dict[layer + '.bias'] = from_numpy(name_weights[key]['bias'])

            def _set_bn(layer, key):
                state_dict[layer + '.running_var'] = from_numpy(name_weights[key]['running_var'])
                state_dict[layer + '.running_mean'] = from_numpy(name_weights[key]['running_mean'])
                state_dict[layer + '.weight'] = torch.ones_like(state_dict[layer + '.running_var'])
                state_dict[layer + '.bias'] = torch.zeros_like(state_dict[layer + '.running_var'])

            def _set_res(layer, n_or_a, index):
                _set_bn(layer+'.bn', n_or_a + 'bn' + str(index))
                _set(layer+'.conv', n_or_a + 'conv' + str(index))
                _set_bn(layer+'.bnr', n_or_a + 'bn' + str(index) + 'r')
                _set(layer+'.convr', n_or_a + 'conv' + str(index) + 'r')

            _set('conv1', 'conv1')
            _set_bn('bn1', 'bn1')
            _set('conv2', 'conv2')
            _set_bn('bn2', 'bn2')
            _set('conv3', 'conv3')
            _set_res('n_res1', 'n', 1)
            _set_res('n_res2', 'n', 2)
            _set_res('n_res3', 'n', 3)
            _set_res('n_res4', 'n', 4)
            _set_res('n_res5', 'n', 5)
            _set_bn('nbn6r', 'nbn6r')
            _set_deconv('nup6', 'nup6')
            _set('nconv6', 'nconv6')
            _set_bn('nbn6', 'nbn6')
            _set('nconv7', 'nconv7')
            _set_bn('nbn7', 'nbn7')
            _set('Nconv0', 'Nconv0')
            _set_res('a_res1', 'a', 1)
            _set_res('a_res2', 'a', 2)
            _set_res('a_res3', 'a', 3)
            _set_res('a_res4', 'a', 4)
            _set_res('a_res5', 'a', 5)
            _set_bn('abn6r', 'abn6r')
            _set_deconv('aup6', 'aup6')
            _set('aconv6', 'aconv6')
            _set_bn('abn6', 'abn6')
            _set('aconv7', 'aconv7')
            _set_bn('abn7', 'abn7')
            _set('Aconv0', 'Aconv0')
            _set('lconv1', 'lconv1')
            _set_bn('lbn1', 'lbn1')
            _set('fc_light', 'fc_light')

            return state_dict

#### FOLLOWING IS SKIP NET IMPLEMENTATION

# Base methods for creating convnet
def get_skipnet_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

def get_skipnet_deconv(in_channels, out_channels, kernel_size=3, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

#versiku
class SkipNet_Encoder(nn.Module):
    def __init__(self):
        super(SkipNet_Encoder, self).__init__()
        #256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        #128
        self.conv2 = get_skipnet_conv(64, 128, kernel_size=4, stride=2, padding=1)
        #64
        self.conv3 = get_skipnet_conv(128, 256, kernel_size=4, stride=2, padding=1)
        #32
        self.conv4 = get_skipnet_conv(256, 512, kernel_size=4, stride=2, padding=1)
        #16
        self.conv5 = get_skipnet_conv(512, 1024, kernel_size=4, stride=2, padding=1)
        #8
        self.conv6 = get_skipnet_conv(1024, 2048, kernel_size=4, stride=2, padding=1)
        #4
        self.fc256 = nn.Linear(32768, 2048)
    
    def get_face(self, sh, normal, albedo):
        shading = get_shading(normal, sh)
        recon   = reconstruct_image(shading, albedo)
        return recon

    def forward(self, x):
        # print('0 ', x.shape )
        out_1 = self.conv1(x)
        # print('1 ', out_1.shape)
        out_2 = self.conv2(out_1)
        # print('2 ', out_2.shape)
        out_3 = self.conv3(out_2)
        # print('3 ', out_3.shape)
        out_4 = self.conv4(out_3)
        # print('4 ', out_4.shape)
        out_5 = self.conv5(out_4)
        # print('5 ', out.shape)
        out = self.conv6(out_5)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        out = self.fc256(out)
        return out, out_1, out_2, out_3, out_4, out_5
        
class SkipNet_Decoder(nn.Module):
    def __init__(self):
        super(SkipNet_Decoder, self).__init__()
        self.dconv1 = get_skipnet_deconv(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.dconv2 = get_skipnet_deconv(1024, 512, kernel_size=4, stride=2, padding=1)
        self.dconv3 = get_skipnet_deconv(512, 256, kernel_size=4, stride=2, padding=1)
        self.dconv4 = get_skipnet_deconv(256, 128, kernel_size=4, stride=2, padding=1)
        self.dconv5 = get_skipnet_deconv(128, 64, kernel_size=4, stride=2, padding=1)
        self.dconv6 = get_skipnet_deconv(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv7  = nn.Conv2d(64, 3, kernel_size=1, stride=1)
    
    def forward(self, x, out_1, out_2, out_3, out_4, out_5 ):
        # print('-0 ', x.shape)
        out = self.dconv1(x)
        # print('-1 ', out.shape, out_4.shape)
        out += out_5
        out = self.dconv2(out)
        # print('-2 ', out.shape, out_3.shape)
        out += out_4
        out = self.dconv3(out)
        # print('-3 ', out.shape, out_2.shape)
        out += out_3
        out = self.dconv4(out)
        # print('-4 ', out.shape, out_1.shape)
        out += out_2
        out = self.dconv5(out)

        out += out_1
        out = self.dconv6(out)
        out = self.conv7(out)

        return out

class SkipNet(nn.Module):
    def __init__(self):
        super(SkipNet, self).__init__()
        self.encoder = SkipNet_Encoder()
        self.normal_mlp = nn.Upsample(scale_factor=4, mode='bilinear')
        self.albedo_mlp = nn.Upsample(scale_factor=4, mode='bilinear')
        self.light_decoder = nn.Linear(2048, 27)
        self.normal_decoder = SkipNet_Decoder()
        self.albedo_decoder = SkipNet_Decoder()
 
    def get_face(self, sh, normal, albedo):
        shading = get_shading(normal, sh)
        recon   = reconstruct_image(shading, albedo)
        return recon
   
    def forward(self, x):
        out, skip_1, skip_2, skip_3, skip_4, skip_5 = self.encoder.forward(x)
        out_mlp = out.unsqueeze(2)
        out_mlp = out_mlp.unsqueeze(3)
        # print(out_mlp.shape, out.shape)
        out_normal = self.normal_mlp(out_mlp)
        out_albedo = self.albedo_mlp(out_mlp)
        # print(out_normal.shape)
        light = self.light_decoder(out)
        normal = self.normal_decoder(out_normal, skip_1, skip_2, skip_3, skip_4,skip_5)

        albedo = self.albedo_decoder(out_albedo, skip_1, skip_2, skip_3, skip_4,skip_5)
     

        #print shading and do recon
        shading = get_shading(normal, light)        
        recon = reconstruct_image(shading, albedo)
        return normal, albedo, light, shading, recon
