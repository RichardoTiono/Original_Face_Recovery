import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms,utils
import torchvision.transforms.functional as F
import glob
import cv2
from random import randint
import os
from skimage import io
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv

from utils import save_image, denorm, get_normal_in_range
import numpy as np
IMAGE_SIZE = 256

def generate_sfsnet_data_csv(dir, save_location):
    albedo = set()
    normal = set()
    depth  = set()
    mask   = set()
    face   = set()
    sh     = set()

    name_to_set = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh}
    
    for k, v in name_to_set.items():
        regex_str = '*/*_' + k + '_*'
        for img in sorted(glob.glob(dir + regex_str)):
            timg = img.split('/')
            folder_id = timg[-2]
            name      = timg[-1].split('.')[0]
            name      = name.split('_')
            assert(len(name) == 4)
            name      = folder_id + '_' + name[0] + '_' + name[2] + '_' + name[3]
            v.add(name)

    final_images = set.intersection(albedo, normal, depth, mask, face, sh)    

    albedo = []
    normal = []
    depth  = []
    mask   = []
    face   = []
    sh     = []
    name   = []

    name_to_list = {'albedo' : albedo, 'normal' : normal, 'depth' : depth, \
                    'mask' : mask, 'face' : face, 'light' : sh, 'name' : name}

    for img in final_images:
        split = img.split('_')
        for k, v in name_to_list.items():
            ext = '.png'
            if k == 'light':
                ext = '.txt'

            if k == 'name':
                filename = split[0] + '_' + split[1] + '_' + k + '_' + '_'.join(split[2:])
            else:
                file_name = split[0] + '/' + split[1] + '_' + k + '_' + '_'.join(split[2:]) + ext
            v.append(file_name)

    df = pd.DataFrame(data=name_to_list)
    df.to_csv(save_location)

def generate_celeba_synthesize_data_csv(dir, save_location):
    albedo = []
    normal = []
    face   = []
    faceHR = []
    sh     = []
    name   = []
    target = []
    attr = []
    #for folder in sorted(glob.glob(dir+ '*/*_albedo*')):

    for img in sorted(glob.glob(dir + '*_albedo*')):
        albedo.append(img)
        
    for img in sorted(glob.glob(dir + '*_normal*')):
        normal.append(img)

    for img in sorted(glob.glob(dir + '*_faceHR*')):
        faceHR.append(img)

    for img in sorted(glob.glob(dir + '*_face.*')):
        face.append(img)
        #no target = face
        target.append(img)
        iname = img.split('/')[-1].split('.')[0]
        name.append(iname)

    for l in sorted(glob.glob(dir + '*_light*')):
        sh.append(l)

    for i in range(len(face)):
        attr.append('0,0,0,1')

    name_to_list = {'albedo' : albedo, 'normal' : normal,'face' : face,'faceHR' : faceHR,'light' : sh, 'name' : name, 'target' : target,'attributes':attr}

    df = pd.DataFrame(data=name_to_list)
    df.to_csv(save_location)
    print('saved')

def generate_synthesize_data_csv(dir, save_location):
    albedo = []
    normal = []
    face   = []
    sh     = []
    name   = []
    target = []
    attr = []

    #for folder in sorted(glob.glob(dir+ '*/*_albedo*')):

    for img in sorted(glob.glob(dir + '*_albedo*')):
        albedo.append(img)
        
    for img in sorted(glob.glob(dir + '*_normal*')):
        normal.append(img)

    for img in sorted(glob.glob(dir + '*_combine*')):
        face.append(img)
        #no target = face
        target.append(img)
        iname = img.split('/')[-1].split('.')[0]
        name.append(iname)

    for l in sorted(glob.glob(dir + '*_light*')):
        sh.append(l)

    for i in range(len(face)):
        attr.append('0,0,0,1')

    name_to_list = {'albedo' : albedo, 'normal' : normal,'face' : face,'light' : sh, 'name' : name, 'target' : target, 'attributes':attr}

    df = pd.DataFrame(data=name_to_list)
    df.to_csv(save_location)
    print('saved')

def generate_celeba_data_csv(dir, save_location):
    face = []
    name = []

    for img in sorted(glob.glob(dir + '*.jpg')):
        face.append(img)
        iname = img.split('/')[-1].split('.')[0]
        name.append(iname)

    face_to_list = {'face': face, 'name':name}
    df = pd.DataFrame(data=face_to_list)
    df.to_csv(save_location)

def generate_rgb_csv(dir, train_subject, save_location):
    face   = []
    train = []
    val = []

    #create psuedo target
    for i in range(2):
        for img in sorted(glob.glob(dir + '/*0_a.png')):
            #2 type of lighting need 2 target
            face.append(img)    
            face.append(img)  

    #loop them together
    #Central light sample
    #red
    for img,target in zip(sorted(glob.glob(dir + '/*0_r*.png')),face):
        iname = img.split('/')[-1].split('.')[0]
        if int(iname.split('_')[0]) <= train_subject:
            train.append([img,iname,target])
        else :
            val.append([img,iname,target])
    #green
    for img,target in zip(sorted(glob.glob(dir + '/*0_g*.png')),face):
        iname = img.split('/')[-1].split('.')[0]
        if int(iname.split('_')[0]) <= train_subject:
            train.append([img,iname,target])
        else :
            val.append([img,iname,target])
    #blue
    for img,target in zip(sorted(glob.glob(dir + '/*0_b*.png')),face):
        iname = img.split('/')[-1].split('.')[0]
        if int(iname.split('_')[0]) <= train_subject:
            train.append([img,iname,target])
        else :
            val.append([img,iname,target])

    #Ambient Sample
    # for i in range(2):
    #     for img in sorted(glob.glob(dir + '/*1_a.png')):
    #         face.append(img)    
    #         face.append(img)  

    # for img,target in zip(sorted(glob.glob(dir + '/*1_r*.png')),face):
    #     iname = img.split('/')[-1].split('.')[0]
    #     red.append([img,iname,target])
    # for img,target in zip(sorted(glob.glob(dir + '/*1_g*.png')),face):
    #     iname = img.split('/')[-1].split('.')[0]
    #     green.append([img,iname,target])
    # for img,target in zip(sorted(glob.glob(dir + '/*1_b*.png')),face):
    #     iname = img.split('/')[-1].split('.')[0]
    #     blue.append([img,iname,target])

    #val set
    df = pd.DataFrame(data=val, columns=['face' ,'name' ,'target'])
    df.to_csv(save_location+'/rgb_test.csv')

    # #training set
    df = pd.DataFrame(data=train,columns=['face' ,'name' ,'target'])
    df.to_csv(save_location+'/rgb_train.csv')


def generate_rgb_real_csv(dir, train_subject,save_location):
    val   = []
    train   = []

    for img,target in zip(sorted(glob.glob(dir + '/*_r*.png')),sorted(glob.glob(dir + '/*_a*.png'))):
        iname = img.split('/')[-1].split('.')[0]
        if int(iname.split('_')[0]) <= train_subject:
            train.append([img,iname,target])
        else :
            val.append([img,iname,target])
    for img,target in zip(sorted(glob.glob(dir + '/*_g*.png')),sorted(glob.glob(dir + '/*_a*.png'))):
        iname = img.split('/')[-1].split('.')[0]
        if int(iname.split('_')[0]) <= train_subject:
            train.append([img,iname,target])
        else :
            val.append([img,iname,target])
    for img,target in zip(sorted(glob.glob(dir + '/*_b*.png')),sorted(glob.glob(dir + '/*_a*.png'))):
        iname = img.split('/')[-1].split('.')[0]
        if int(iname.split('_')[0]) <= train_subject:
            train.append([img,iname,target])
        else :
            val.append([img,iname,target])

    #val set
    df = pd.DataFrame(data=val, columns=['face' ,'name' ,'target'])
    df.to_csv(save_location+'/rgb_real_test.csv')

    #train set
    df = pd.DataFrame(data=train,columns=['face' ,'name' ,'target'])
    df.to_csv(save_location+'/rgb_real_train.csv')

def get_sfsnet_dataset(syn_dir=None, read_from_csv=None, read_celeba_csv=None, read_first=None, validation_split=0, training_syn=False):
    albedo  = []
    sh      = []
    normal  = []
    face    = []
    faceHR = []

    if training_syn:
        read_celeba_csv = None
        read_rgb_csv = None

    if read_from_csv is None:
        for img in sorted(glob.glob(syn_dir + '*/*_albedo*')):
            albedo.append(img)
        for img in sorted(glob.glob(syn_dir + '*/*_combine*')):
            face.append(img)    

        for img in sorted(glob.glob(syn_dir + '*/*_normal*')):
            normal.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_light*.txt')):
            sh.append(img)
    else:

        # df = pd.read_csv(read_from_csv)
        # df = df[:read_first]
        # #debugging dikasih batas 5000 training data
        # albedo = list(df['albedo'])
        # face   = list(df['face'])
        # faceHR   = list(df['face'])
        # normal = list(df['normal'])
        # sh     = list(df['light'])

        name_to_list = {'albedo' : albedo, 'normal' : normal,'face' : face, 'light' : sh, 'faceHR' : faceHR}

        for _, v in name_to_list.items():
            #v[:] = [syn_dir + el for el in v]
            v[:] = [el for el in v]
        # Merge Synthesized Celeba dataset for Psedo-Supervised training
        if read_celeba_csv is not None:
            df = pd.read_csv(read_celeba_csv)
            df = df[:read_first]
            albedo += list(df['albedo'])
            face   += list(df['face'])
            faceHR  += list(df['faceHR'])
            normal += list(df['normal'])
            sh     += list(df['light'])

    assert(len(albedo) == len(face) == len(normal) == len(sh)== len(faceHR))
    dataset_size = len(albedo)
    validation_count = int (math.ceil(validation_split * dataset_size / 100))
    train_count      = dataset_size - validation_count
    # Build custom datasets
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor()
            ])
    full_dataset = SfSNetDataset(albedo, face,faceHR, normal, sh , transform)  
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

def get_celeba_dataset(dir=None, read_from_csv=None, read_first=None, validation_split=0):
    face    = []

    if read_from_csv is None:
        for img in sorted(glob.glob(dir + '/*.jpg')):
            face.append(img)    
        if len(face)==0:
            for img in sorted(glob.glob(dir + '/*.png')):
                face.append(img)    
    else:
        df = pd.read_csv(read_from_csv)
        df = df[:read_first]
        face   = list(df['face'])
    dataset_size = len(face)
    validation_count = int (math.ceil(validation_split * dataset_size / 100))
    train_count      = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor()
                ])
    full_dataset = CelebADataset(face, transform)  
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

def get_sample_dataset(dir=None, read_from_csv=None, read_first=None, validation_split=0):
    face    = []
    name = []
    if read_from_csv is None:
        for img in sorted(glob.glob(dir + '/*.jpg')):
            face.append(img) 
            name.append(os.path.basename(img))   
        if len(face)==0:
            for img in sorted(glob.glob(dir + '/*.png')):
                face.append(img)    
                name.append(os.path.splitext(os.path.basename(img))[0]) 
    dataset_size = len(face)
    validation_count = int (math.ceil(validation_split * dataset_size / 100))
    train_count      = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor()
                ])
    full_dataset = CelebADataset(face, transform, name)  
    return full_dataset


def get_light_dataset(gen_dir = None, target_dir=None, read_rgb_synt_csv=None,read_rgb_real_csv=None, read_first=None, validation_split=0, training_syn=False):
    face    = []
    target = []
    attr = []

    
    if gen_dir is not None:
        for img in sorted(glob.glob(gen_dir + '/*.png')):
            face.append(img)  
        for img in sorted(glob.glob(target_dir + '/*.png')):
            target.append(img)    
    else:
        df = pd.read_csv(read_rgb_synt_csv)
        df = df[:(read_first)]
        face   = list(df['face'])
        target     = list(df['target'])


        name_to_list = {'face' : face, 'target' : target, 'attributes': attr}

        for _, v in name_to_list.items():
            #v[:] = [syn_dir + el for el in v]
            v[:] = [el for el in v]
        # Merge Real data with synthesized data
        if read_rgb_real_csv is not None:
            df = pd.read_csv(read_rgb_real_csv)
            df = df[:(read_first)]
            face   += list(df['face'])
            target     += list(df['target'])


    dataset_size = len(face)
    validation_count = int (math.ceil(validation_split * dataset_size / 100))
    train_count      = dataset_size - validation_count

    full_dataset = LightDataset(face =face, attr=attr, target=target) 
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

def generate_celeba_synthesize(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None):
 
    # debugging flag to dump image
    fix_bix_dump = 0
    recon_loss  = nn.L1Loss() 

    if use_cuda:
        recon_loss  = recon_loss.cuda()

    tloss = 0 # Total loss
    rloss = 0 # Reconstruction loss

    for bix, data in enumerate(dl):
        face,faceHR = data
        if use_cuda:
            face   = face.cuda()
            faceHR = faceHR.cuda()

        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face,_,_ = sfs_net_model(face)
        for i in range(face.size(0)):
            file_name = out_folder + str(train_epoch_num) + '_' + str(bix)+'_'+str(i)
            # log images
            predicted_normal = denorm(predicted_normal)
            save_image(predicted_normal[i], path = file_name+'_normal.png')
            save_image(predicted_albedo[i], path = file_name+'_albedo.png')
            save_image(predicted_shading[i], path = file_name+'_shading.png')
            save_image(predicted_face[i], path = file_name+'_recon.png')
            save_image(face[i], path = file_name+'_face.png')
            save_image(faceHR[i], path = file_name+'_faceHR.png')
            np.savetxt(file_name+'_light.txt', predicted_sh[i].cpu().detach().numpy(), delimiter='\t')


def generate_celeba_sample(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None):

        for bix, data in enumerate(dl):
            face = data
            if use_cuda:
                face   = face.cuda()

            # predicted_face == reconstruction
            predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(face)
            #save predictions in log folder
            file_name = out_folder + str(train_epoch_num) + '_' + str(bix)+'.png'

            # log images
            predicted_normal = denorm(predicted_normal)
            tensor = torch.cat([predicted_albedo, predicted_normal,predicted_shading,predicted_face,face], dim=0)
            utils.save_image(tensor.cpu(), file_name , nrow=5)
            # save_image(face, path = file_name+'_face.png')
            # save_image(predicted_normal, path = file_name+'_normal.png')
            # save_image(predicted_albedo, path = file_name+'_albedo.png')
            # save_image(predicted_shading, path = file_name+'_shading.png')
            # save_image(predicted_face, path = file_name+'_recon.png')
            np.savetxt(file_name+'_light.txt', predicted_sh.cpu().detach().numpy(), delimiter='\t')


def generate_rgb(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None):
 

    for bix, data in enumerate(dl):
        face,r,g,b = data
        if use_cuda:
            face   = face.cuda()

        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(face)
        #save predictions in log folder
        file_name = out_folder + str(train_epoch_num) +'_' + str(bix)

        # log images
        predicted_normal = denorm(predicted_normal)
        save_image(r, path = file_name+'_r.png')
        save_image(g, path = file_name+'_g.png')
        save_image(b, path = file_name+'_b.png')
        save_image(predicted_normal, path = file_name+'_normal.png')
        save_image(predicted_albedo, path = file_name+'_albedo.png')
        save_image(predicted_shading, path = file_name+'_shading.png')
        save_image(face, path = file_name+'_target.png')
        np.savetxt(file_name+'_light.txt', predicted_sh.cpu().detach().numpy(), delimiter='\t')

def visualize_activation(model,dl,output_folder,use_cuda):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for bix, data in enumerate(dl):
        face,_ = data
        if use_cuda:
            face   = face.cuda()

        for name, module in model.named_modules():
            #print(name,module)
            if isinstance(module, nn.ConvTranspose2d):
                module.register_forward_hook(get_activation(name))
        #print(a)
        predicted_face = model(face)
        
        for name in activation:
            act = activation[name].squeeze()
            num_plot = 2
            row = 2
            col = np.ceil(num_plot/row)
            fig= plt.figure(1)
            for idx in range(num_plot):
                ax = fig.add_subplot(row,col,1+idx)
                ax.imshow(act[idx].cpu())
                ax.set_axis_off()
            plt.savefig(output_folder+str(bix)+'_'+name+'.png', bbox_inches='tight')



def generate_light_extractor_sample(generator, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None):

        for bix, data in enumerate(dl):
            face,name = data
            if use_cuda:
                face   = face.cuda()
            # predicted_face == reconstruction
            predicted_face= generator(face)
            #predicted_combination= predicted_face + predicted_color
            #save predictions in log folder
            file_name = out_folder+str(name[0])

            # log images
            #save_image(face, path = file_name+'_real.png')
            save_image(predicted_face, path = file_name+'_fake.png')
            # tensor = torch.cat([face,predicted_face], dim=0)
            # utils.save_image(tensor.cpu(), file_name+".png" , nrow=2)
            #np.savetxt(file_name+'_light.txt', label.cpu().detach().numpy(), delimiter='\t')


class SfSNetDataset(Dataset):
    def __init__(self, albedo, face, faceHR, normal, sh, transform = None):
        self.albedo = albedo
        self.face   = face
        self.faceHR   = faceHR
        self.normal = normal
        self.sh     = sh
        self.transform = transform
        self.dataset_len = len(self.albedo)
        self.normal_transform = transforms.Compose([
                              transforms.Resize(256)
                            ])

    def __getitem__(self, index):
        albedo = self.transform(Image.open(self.albedo[index]))
        face   = self.transform(Image.open(self.face[index]))
        faceHR   = self.transform(Image.open(self.faceHR[index]))
        # normal = io.imread(self.face[index]))
        normal = self.normal_transform(Image.open(self.normal[index]))
        normal = torch.tensor(np.asarray(normal)).permute([2, 0, 1])
        normal = normal.type(torch.float)
        normal = (normal - 128) / 128
        #normal = denorm(normal)
        pd_sh  = pd.read_csv(self.sh[index], sep='\t', header = None)
        sh     = torch.tensor(pd_sh.values).type(torch.float).reshape(-1)
        return albedo, normal, sh, face,faceHR

    def __len__(self):
        return self.dataset_len

class LightDataset(Dataset):
    def __init__(self, albedo=None, face=None, normal=None, sh=None ,attr=None, target = None):
        self.albedo = albedo
        self.face   = face
        self.normal = normal
        self.sh     = sh
        self.attr = attr
        self.target = target
        self.dataset_len = len(self.face)

    def convert(self,image):
            image.load() # required for png.split()
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1]) # 3 is the alpha channel
            return background

    def transform(self, image, target):

        resize = transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE))
        image = resize(image)
        target = resize(target)
        # # Random crop
        # random_angle = randint(0,360)
        # image = F.rotate(image, random_angle)
        # target = F.rotate(target, random_angle)

        # # Random horizontal flipping
        # if randint(0,100) > 50:
        #     image = F.hflip(image)
        #     target = F.hflip(target)

        # # Random vertical flipping
        # if randint(0,100) > 50:
        #     image = F.vflip(image)
        #     target = F.vflip(target)

        # Transform to tensor
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target

    def __getitem__(self, index):
        face   = Image.open(self.face[index])
        target   = Image.open(self.target[index])
        if(face.mode=='RGBA'):
            face= self.convert(face)
        if(target.mode=='RGBA'):
            target= self.convert(target)
        face,target   = self.transform(face,target)
        if len(self.attr)!=0:
            labels = [0 if i=='0' else 1 for i in self.attr[index].split(',')]
            attr = torch.tensor(labels).type(torch.float).reshape(-1)
            return attr, face, target
        else :
            return face, target

    def __len__(self):
        return self.dataset_len



class CelebADataset(Dataset):
    def __init__(self, face, transform = None, name = None):
        self.face   = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.name = name

    def __getitem__(self, index):

        png = Image.open(self.face[index])
        if(png.mode=='RGBA'):
            png.load() # required for png.split()
            background = Image.new("RGB", png.size, (255, 255, 255))
            background.paste(png, mask=png.split()[-1]) # 3 is the alpha channel
            face   = self.transform(background)
        else:
            face   = self.transform(png)

        if self.name is not None :
            name = self.name[index]
            return face,name
        else :
            return face

    def __len__(self):
        return self.dataset_len


class RGBDataset(Dataset):
    def __init__(self, face,r,g,b, transform = None,):
        self.face   = face
        self.r      = r
        self.g      = g
        self.b      = b
        self.transform = transform
        self.dataset_len = len(self.face)

    def convert(self,image):
            image.load() # required for png.split()
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1]) # 3 is the alpha channel
            return background

    def __getitem__(self, index):
        face   = Image.open(self.face[index])
        r   = Image.open(self.r[index])
        g   = Image.open(self.g[index])
        b   = Image.open(self.b[index])
        if(face.mode=='RGBA'):
            face= self.convert(face)
        if(r.mode=='RGBA'):
            r=self.convert(r)
        if(g.mode=='RGBA'):
            g=self.convert(g)
        if(b.mode=='RGBA'):
            b=self.convert(b)
        face   = self.transform(face)
        r   = self.transform(r)
        g   = self.transform(g)
        b   = self.transform(b)
        return face,r,g,b

    def __len__(self):
        return self.dataset_len