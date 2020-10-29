import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms,utils
import torchvision.transforms.functional as F
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
from function.arch_light import Generator
import dlib
import cv2
import numpy as np 
from imutils import face_utils


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

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
    jawline[17:]=np.flip(jawline,axis=0)[0:11]
    jawline=np.asarray(jawline,dtype=dtype)

    return jawline

def getmask(img,jawline):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.asarray(img)
    # create mask
    polygon = jawline.flatten().tolist()
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon,  fill='white')    
    mask = np.array(maskIm)
    return mask

def createmask (image):
    predictor_path = './dlib/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path) 
    

    img= np.array(image)
    rects = detector(img, 1)
    for (i, rect) in enumerate(rects):
            bounding = face_utils.rect_to_bb(rect)
            shape = predictor(img, rect)
            jawline = getmaskcoord(shape)
            mask = getmask(img,jawline)
    return mask,bounding

def generate_light_extractor_sample(generator, face_tensor):

    face = face_tensor
    if torch.cuda.is_available():
        face   = face.cuda()

    # predicted_face == reconstruction

    predicted_face= generator(face)
    return face,predicted_face

def SampleDataset(face,transform):
    png = Image.open(face)
    new_img = png.copy()
    if(png.mode=='RGBA'):
        png.load() # required for png.split()
        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[-1]) # 3 is the alpha channel

    mask, bounding = createmask(new_img)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask= np.float32(mask)/255  
    img = np.multiply(new_img,mask)
    img = np.uint8(img)
    #Cropping
    x, y, w, h = bounding
    #print(x, y, w, h)
    if w < 256 :
        x = x-10
        w = 256
    if h <256 :
        y = y -10
        h = 256
    if (x >= 0) and (y >= 0) and (w<=256) and (h<=256):
        img = img[y:y+h,x:x+w,:]
    face   = transform(img)
    face = torch.unsqueeze(face, 0)
    return face



def transform_data(face_path):
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.ToTensor()
                ])
    face_tensor = SampleDataset(face_path, transform) 
    return face_tensor

def main(face_path):
    generator = Generator()
    generator.eval()
    if torch.cuda.is_available():
        print("Number of GPU used :", torch.cuda.device_count(), "GPUs!")
        generator = nn.DataParallel(generator)
        generator = generator.cuda()

    print("using pretrained model")
    checkpoint = torch.load('./model/light_gan10.pkl')
    generator.load_state_dict(checkpoint['generator_state_dict'])

    face_tensor = transform_data(face_path)


    face,result_face = generate_light_extractor_sample(generator,face_tensor)
    return face,result_face

    