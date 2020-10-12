from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
from PIL import Image, ImageDraw
import ntpath
import os
import matplotlib.image as mpimg


# Code to Create Mask

def rect_to_bb(rect):
    #create Bounding Box around the face
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y


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
    ImageDraw.Draw(maskIm).polygon(polygon, fill='white')
      
    mask = np.array(maskIm)
    return mask


#DLIB Library
detector = dlib.get_frontal_face_detector()
list_im=[f for f in glob.glob('/home/cgal/testing/black_bg/batchnorm+skip/*.png')]
predictor_path = '/home/cgal/dlib/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for f in list_im:
    
    img = dlib.load_rgb_image(f)

    #Detect and return the face position
    rects = detector(img, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        jawline = getmaskcoord(shape)
    mask = getmask(img,jawline)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)/255
    facemask=np.float32(np.multiply(mask,img))/255
    name=os.path.splitext(ntpath.basename(f))[0]
    name="/"+name
    dirname=os.path.dirname(f)+'/masking'
    os.system('mkdir -p {}'.format(dirname))

    fullname=dirname + name+'.png'
    mpimg.imsave(fullname, mask)





