import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
#from utils import denorm
import numpy as np

def get_shading(N, L):
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
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

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
        # self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=2)
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
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
        # self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=2)
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
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
        self.pool  = nn.AvgPool2d(128) 
        self.fc    = nn.Linear(128, 27)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        # reshape to batch_size x 128
        out = out.view(-1, 128)
        out = self.fc(out)
        return out

def reconstruct_image(shading, albedo):
    return shading * albedo
        
class AlbedoCorrector(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """
    def __init__(self):
        super(AlbedoCorrector, self).__init__()
        self.conv1 = get_conv(3, 32, kernel_size=3, stride=1, padding = 1)
        self.conv2  = get_conv(32, 64, kernel_size=3, stride=1, padding = 1)
        self.conv3  = get_conv(64, 64, kernel_size=3, stride=1, padding = 1)
        self.conv4  = get_conv(64, 32, kernel_size=3, stride=1, padding = 1)
        self.conv5  = get_conv(32, 3, kernel_size=3, stride=1, padding = 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out+x
        return out



class ShadingCorrector(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """
    def __init__(self):
        super(ShadingCorrector, self).__init__()
        self.conv1 = get_conv(3, 32, kernel_size=3, stride=1, padding = 1)
        self.conv2  = get_conv(32, 64, kernel_size=3, stride=1, padding = 1)
        self.conv3  = get_conv(64, 64, kernel_size=3, stride=1, padding = 1)
        self.conv4  = get_conv(64, 32, kernel_size=3, stride=1, padding = 1)
        self.conv5  = get_conv(32, 3, kernel_size=3, stride=1, padding = 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out+x
        return out

    #     self.encoder = Encoder()
    #     #self.bottleneck= Bottleneck(256,256)
    #     self.decoder_img = Decoder_img()

    # def forward(self, x):
    #     out = self.encoder(x)
    #     #out = self.bottleneck(out)
    #     out_img = self.decoder_img(out)
    #     out =  x + out_img
    #     return out

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

