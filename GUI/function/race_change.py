import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import glob
import random 

def check_distance(ori,target):
    h,w = ori.shape
    x_mapping = np.zeros((w,h))
    y_mapping = np.zeros((w,h))
    for i in range(w-1):
        for j in range(h-1):
            #cek distance
            substract_result = np.abs(target-ori.item(i,j))
            #skip pengambilan sample tiap 30x30 
            #find min distance index
            index = np.where(substract_result == substract_result.min())
            #record to mapping
            x_mapping[i,j] = index[0][0]
            y_mapping[i,j] = index[1][0]
    return x_mapping,y_mapping

def do_full_intensities(original,target):
    
    h,w,c = original.shape
    hue = np.zeros((w,h,1))
    saturation = np.zeros((w,h,1))
    val = np.zeros((w,h,1))
    tar = target[:,:,2].copy()
    ori = original[:,:,2].copy()
    
    target = cv2.cvtColor(target,cv2.COLOR_HSV2BGR_FULL)

    
    #get mean and std
    mean_target = target.mean()
    std_target = target.std()
    mean_ori = ori.mean()
    std_ori = ori.std()
    
    #Change target intensities to image input
    tar = np.float32(tar)
    tar -= mean_target
    tar *= std_ori/std_target
    tar += mean_ori
    tar = tar.astype(int)
    
    x_mapping,y_mapping = check_distance(ori, tar)
        
    for i in range(x_mapping.shape[0]):
        for j in range(x_mapping.shape[1]):
#             hue[i][j]= target[:,:,0].item(int(x_mapping[i][j]),int(y_mapping[i][j]))
#             saturation[i][j]= target[:,:,1].item(int(x_mapping[i][j]),int(y_mapping[i][j]))
            hue[i][j]= target[:,:,0].item(int(x_mapping[i][j]),int(y_mapping[i][j]))
            saturation[i][j]= target[:,:,1].item(int(x_mapping[i][j]),int(y_mapping[i][j]))
            val[i][j]= target[:,:,2].item(int(x_mapping[i][j]),int(y_mapping[i][j]))

    #new_image = np.concatenate((hue,saturation,ori[:,:,np.newaxis]),axis=2)
    new_image = np.concatenate((hue,saturation,val),axis=2)
    
    new_image = np.uint8(new_image)
    
    return new_image

def getmaskcoord(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    #get mouth
    mouth = []
    #gigi hilang
    for i in range(48,59):
    #gigi muncul
    #for i in range(48,68):
        mouth.append((shape.part(i).x, shape.part(i).y))
    
    mouth=np.asarray(mouth,dtype=dtype)
    # return the list of (x, y)-coordinates
    return mouth


def createmask (image_path):
    predictor_path = './dlib/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path) 
    
    img = dlib.load_rgb_image(image_path)    
    rects = detector(img, 1)
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(img, rects[0])
    mouth = getmaskcoord(shape)
                       
    return mouth


def main(path,choice):
    image = cv2.imread (path)
    if (choice == "African"):
        tar_image= cv2.imread('./skin_reference/black-ori.png')
    elif (choice == "American-pink"):
        tar_image= cv2.imread('./skin_reference/pink-ori.png')
    elif (choice == "American-white"):
        tar_image= cv2.imread('./skin_reference/white-ori.png')

    #convert RGB
    image = cv2.resize(image,(256,256))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_target = cv2.cvtColor(tar_image,cv2.COLOR_BGR2RGB)

    #transfer = color_transfer(tar_image, image,  clip='t', preserve_paper='t')

    ori_image = image.copy()
    tar_image = image_target.copy()

    #change to HSV
    ori_image = cv2.cvtColor(ori_image,cv2.COLOR_RGB2HSV_FULL)
    tar_image = cv2.cvtColor(tar_image,cv2.COLOR_RGB2HSV_FULL)


    # std_hue_ori,mean_hue_ori = ori_image[:,:,0].std(),ori_image[:,:,0].mean()
    # std_sat_ori,mean_sat_ori = ori_image[:,:,1].std(),ori_image[:,:,1].mean()
    # std_val_ori,mean_val_ori = ori_image[:,:,2].std(),ori_image[:,:,2].mean()
    # std_hue_tar,mean_hue_tar = tar_image[:,:,0].std(),tar_image[:,:,0].mean()
    # std_sat_tar,mean_sat_tar = tar_image[:,:,1].std(),tar_image[:,:,1].mean()
    # std_val_tar,mean_val_tar = tar_image[:,:,2].std(),tar_image[:,:,2].mean()

    #First Method (Color Transfer Paper)
    # ori_image[:,:,0] = do_math(ori_image[:,:,0], tar_image[:,:,0] ,std_hue_ori,mean_hue_ori,std_hue_tar,mean_hue_tar)
    # ori_image[:,:,1] = do_math(ori_image[:,:,1], tar_image[:,:,1] ,soriginal[:,:,1][:,:,np.newaxis]td_sat_ori,mean_sat_ori,std_sat_tar,mean_sat_tar)
    # ori_image[:,:,2] = do_math(ori_image[:,:,2], tar_image[:,:,2] ,std_val_ori,mean_val_ori,std_val_tar,mean_val_tar)

    #Second Method (Toward Race-related race detection)
    new_image = do_full_intensities(ori_image,tar_image)


    #change back to RGB
    #new_image = cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB_FULL)
    tar_image = cv2.cvtColor(tar_image,cv2.COLOR_HSV2RGB_FULL)
    return new_image