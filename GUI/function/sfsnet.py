#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import glob
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
from imutils import face_utils
import argparse
import dlib
from function.arch_sfsnet import SfsNetPipeline
from PIL import Image, ImageDraw
import torch.nn as nn
from torchvision import transforms
from function.color_harmonize import main as harmonize
from function.skin_change import main as race


# In[2]:


def lambertian_attenuation(n) :
#%a = [.8862; 1.0233; .4954];
    pi = 3.14
    a = pi*np.array([1,2/3,0.25])
    if n > 3:
        print('didnt record more than 3 attenuations')
    o = a[0:n]
    return o


# In[3]:


def normal_harmonics(N, att):
    # % Return the harmonics evaluated at surface normals N, attenuated by att.
    # % Normals can be scaled surface normals, in which case value of each
    # % harmonic at each point is scaled by albedo.
    # % Harmonics written as polynomials
    # % 0,0    1/sqrt(4*pi)
    # % 1,0    z*sqrt(3/(4*pi))
    # % 1,1e    x*sqrt(3/(4*pi))
    # % 1,1o    y*sqrt(3/(4*pi))
    # % 2,0   (2*z.^2 - x.^2 - y.^2)/2 * sqrt(5/(4*pi))
    # % 2,1e  x*z * 3*sqrt(5/(12*pi))
    # % 2,1o  y*z * 3*sqrt(5/(12*pi))
    # % 2,2e  (x.^2-y.^2) * 3*sqrt(5/(48*pi))
    # % 2,2o  x*y * 3*sqrt(5/(12*pi))
    pi = 3.14
    xs = (N[:,0])
    ys = (N[:,1])
    zs = (N[:,2])
    a = np.sqrt(pow(xs,2)+pow(ys,2)+pow(zs,2))
    #denom = (a==0) + a
    #x = xs./a; y = ys./a; z = zs./a;
    x = xs / a
    y = ys / a
    z = zs / a

    x2 = np.multiply(x,x)
    y2 = np.multiply(y,y)
    z2 = np.multiply(z,z)
    xy = np.multiply(x,y)
    xz = np.multiply(x,z)
    yz = np.multiply(y,z)
    
    H1 = att[0]*(1/np.sqrt(4*pi)) * a
    H2 = att[1]*(np.sqrt(3/(4*pi))) * zs
    H3 = att[1]*(np.sqrt(3/(4*pi))) * xs
    H4 = att[1]*(np.sqrt(3/(4*pi))) * ys
    H5 = att[2]*(1/2)*(np.sqrt(5/(4*pi))) * (np.multiply((2*z2 - x2 - y2) , a))
    H6 = att[2]*(3*np.sqrt(5/(12*pi))) * (np.multiply(xz , a))
    H7 = att[2]*(3*np.sqrt(5/(12*pi))) * (np.multiply(yz , a))
    H8 = att[2]*(3*np.sqrt(5/(48*pi))) * (np.multiply((x2 - y2) , a))
    H9 = att[2]*(3*np.sqrt(5/(12*pi))) *(np.multiply(xy , a))
    H = [H1,H2,H3,H4,H5,H6,H7,H8,H9]
    return H

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


# In[4]:


def create_shading_recon(n_out2,al_out2,light_out):
    M=n_out2.shape[0]
    N=n_out2.shape[1]
    No1=np.reshape(n_out2,(M*N,3),order='F')
    #tex1=al_out2.transpose(2,1,0).reshape(M*M,3)
    
    la = lambertian_attenuation(3)
    HN1 = normal_harmonics(No1, la)
    
    HN1 = np.array(HN1).transpose()
    HS1r=np.dot(HN1,light_out[0:9])
    HS1g=np.dot(HN1,light_out[9:18]) 
    HS1b=np.dot(HN1,light_out[18:27]) 
    HS1r=np.reshape(HS1r,[1,M ,N],order='F')
    HS1g=np.reshape(HS1g,[1,M ,N],order='F') 
    HS1b=np.reshape(HS1b,[1,M ,N],order='F') 
    HS1=np.concatenate((HS1r,HS1g,HS1b),axis = 0)
    HS1=np.moveaxis(HS1, 0, -1)
    Tex1=np.multiply(al_out2,HS1)
    
    IRen0=Tex1
    Shd=HS1*(200/255) #200 is added instead of 255 so that not to scale the shading to all white
    Ishd0=Shd
    return IRen0,Ishd0


# ## Face Detection


def getmaskcoord(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    jawline = []
    #get the jawline
    for i in range(0, 27):
        if(i>16):
            jawline.append((shape.part(i).x, shape.part(i).y-10))
        else:
            jawline.append((shape.part(i).x, shape.part(i).y))
        if(i==21):
            jawline.append((shape.part(27).x, shape.part(19).y))
#     jawline.append((shape.part(17).x, shape.part(19).y-5))
#     jawline.append((shape.part(26).x, shape.part(19).y-5))
    jawline[17:]=np.flip(jawline,axis=0)[0:11]
    jawline=np.asarray(jawline,dtype=dtype)
    #get eyes
    eyes = []
    for i in range(36,48):
        eyes.append((shape.part(i).x, shape.part(i).y))
        
    eyes=np.asarray(eyes,dtype=dtype)
    #get mouth
    mouth = []
    #gigi hilang
    for i in range(48,59):
    #gigi muncul
    #for i in range(48,68):
        mouth.append((shape.part(i).x, shape.part(i).y))
    
    mouth=np.asarray(mouth,dtype=dtype)
    # return the list of (x, y)-coordinates
    return jawline,eyes,mouth


# In[7]:


def getmask(img,jawline):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.asarray(img)
    # create mask
    polygon = jawline.flatten().tolist()
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon,  fill='white')
    #ImageDraw.Draw(maskIm).polygon(polygon, outline=(1))
    # draw eyes
    # righteyes=eyes[0:6].flatten().tolist()
    # ImageDraw.Draw(maskIm).polygon(righteyes,  fill='black')
    # lefteyes=eyes[6:].flatten().tolist()
    # ImageDraw.Draw(maskIm).polygon(lefteyes, fill='black')
    # # draw mouth
    # mouth=mouth.flatten().tolist()
    # ImageDraw.Draw(maskIm).polygon(mouth, fill='black')
    
    mask = np.array(maskIm)
    return mask


# In[8]:


def getface(img,jawline,eyes,mouth):
    im=img.copy()
    img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    imArray = np.asarray(img)
    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), color=255)
    #create jawline
    jaw = jawline.flatten().tolist()
    ImageDraw.Draw(maskIm).polygon(jaw,  fill='black')
    #draw eyes
    righteyes=eyes[0:6].flatten().tolist()
    ImageDraw.Draw(maskIm).polygon(righteyes, fill='white')
    lefteyes=eyes[6:].flatten().tolist()
    ImageDraw.Draw(maskIm).polygon(lefteyes, fill='white')
    # draw mouth
    mouth=mouth.flatten().tolist()
    ImageDraw.Draw(maskIm).polygon(mouth, fill='white')
    mask = np.array(maskIm)
    cutmask = cv2.bitwise_or(im,im,mask=mask)
    return mask


# In[9]:


def createmask (image):
    predictor_path = './dlib/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path) 
    
    # if isinstance(image_path, str): 
    #     img = dlib.load_rgb_image(image_path)   
    # else :
    img= np.array(image)
    rects = detector(img, 1)
    #for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
    shape = predictor(img, rects[0])
    jawline,eyes,mouth = getmaskcoord(shape)
    face_mask = getmask(img,jawline)
    facepart_mask= getface(img,jawline,eyes,mouth)
            
    return face_mask,facepart_mask

#main part
def main(img, mask ,template = None, degree = None , skin = None):
    M = 256
    P = 256
    L = 256
    
    net = SfsNetPipeline()
    if torch.cuda.is_available():
        net = nn.DataParallel(net)
        net = net.cuda()
    net.eval()
    checkpoints=torch.load('./model/sfs_net_model_9.pkl')
    net.load_state_dict(checkpoints['model_state_dict'],strict = False)

    if isinstance(img, str): 
        img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(M,M))
    face_mask,_=createmask(img)
    img = np.float32(img)/255
    #facepart_mask=cv2.cvtColor(facepart_mask, cv2.COLOR_GRAY2RGB)
    face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
    face_mask=np.float32(face_mask)/255  
    #facepart_mask=np.float32(facepart_mask)/255  

    face_mask=cv2.resize(face_mask,(P,L))
    #facepart_mask=cv2.resize(facepart_mask,(P,L))

    #create mask for input image
    #facepart=np.multiply(img,facepart_mask) + np.multiply((1-facepart_mask),np.ones((L,P,3)))
    if (mask == True):
        img = np.multiply(img,face_mask)
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((P,L)),
            transforms.ToTensor()
            ])
    img *=255
    img = np.uint8(img)
    im_input = transform(img)
    im_input = im_input[np.newaxis,:,:,:]
    a=im_input[0].view(3,256,256).permute(1,2,0)
    n, a, l, sh,t,a_new,sh_new,t_new = net(im_input)
    al_out = a.detach().cpu().numpy()

    #change to h,w,c
    al_out = np.squeeze(al_out, 0)
    al_out=np.moveaxis(al_out, 0, -1)

    #normalize
    #n_out = (n_out + 1) / 2


    #%creates reconstruction and shading image
    al_out*=255
    al_out[al_out>255]= 255
    al_out[al_out<0]=0
    al_out = np.uint8(al_out)


    # #changeskin part
    # if skin is not None :
    #     pref = 0
    #     if skin == "Caucasian":
    #         pref = 1
    #     al_out = race(al_out, pref = pref)

    # # mask=cv2.resize(mask,(P,L))
    # # mask2=cv2.resize(mask2,(P,L))
    # #uncomment untuk melakukan color harmonize
    # if template is not None :
    #     al_out = harmonize(al_out,degree = degree, template= template)

    al_out=transform(al_out).cuda()
    al_out = al_out[np.newaxis,:]


    Ishd=get_shading(n,l) 

    # Ishd = Ishd.detach().cpu().numpy()
    # Ishd = np.squeeze(Ishd, 0)
    # Ishd = np.moveaxis(Ishd, 0, -1)
    # Ishd *= 255
    # Ishd = np.clip (Ishd,0,255)
    # Ishd = np.uint8(Ishd)
    # Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2GRAY)
    # Ishd = cv2.cvtColor(Ishd, cv2.COLOR_GRAY2RGB)
    # Ishd = transform(Ishd).cuda()
    # Ishd = Ishd[np.newaxis,:]

    # a=Ishd[0].view(3,256,256).permute(1,2,0)
    # a=a.detach().cpu().numpy()
    # b=sh[0].view(3,256,256).permute(1,2,0)
    # b=b.detach().cpu().numpy()
    # fig=plt.figure(figsize=(3, 3))
    # fig.add_subplot(131)
    # plt.imshow(a)
    # fig.add_subplot(132)
    # plt.imshow(b)          
    # plt.show()

    #replace albedo
    # al_out = cv2.imread("/home/cgal/al.png")
    # al_out = cv2.cvtColor(al_out, cv2.COLOR_BGR2RGB)
    # al_out = transform(al_out).cuda()
    # al_out = al_out[np.newaxis,:]


    Irec = al_out*Ishd

    im_input = im_input.detach().cpu().numpy()
    n_out = n.detach().cpu().numpy()
    light_out = l.detach().cpu().numpy()
    Irec = Irec.detach().cpu().numpy()
    Ishd = Ishd.detach().cpu().numpy()
    al_out = al_out.detach().cpu().numpy()


    al_out = np.squeeze(al_out, 0)
    al_out=np.moveaxis(al_out, 0, -1)
    n_out = np.squeeze(n_out, 0)
    n_out=np.moveaxis(n_out, 0, -1)
    light_out =  np.moveaxis(light_out, 0, -1)
    Ishd = np.squeeze(Ishd, 0)
    Ishd=np.moveaxis(Ishd, 0, -1)
    Irec = np.squeeze(Irec, 0)
    Irec=np.moveaxis(Irec, 0, -1)
    
    im_input = np.squeeze(im_input, 0)
    im_input=np.moveaxis(im_input, 0, -1)
    im_input = cv2.cvtColor(im_input, cv2.COLOR_BGR2RGB)
    

    n_out = (n_out + 1) / 2

    image=im_input
    # normal = np.multiply(n_out,face_mask)+ np.multiply((1-face_mask),np.ones((L,P,3)))
    # albedo=np.multiply(al_out,face_mask)+ np.multiply((1-face_mask),np.ones((L,P,3)))
    # shading = np.multiply(Ishd,face_mask)+ np.multiply((1-face_mask),np.ones((L,P,3)))
    # recon=np.multiply(Irec,face_mask)+ np.multiply((1-face_mask),np.ones((L,P,3)))
    normal = n_out.copy()
    albedo=al_out.copy()
    shading = Ishd.copy()
    recon=Irec.copy()
    return image,normal,albedo,shading,recon





