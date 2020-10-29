import cv2
import numpy as np


def changeskincolor(image,tipe,satlevel,vallevel):
    ori_image=image.copy()
    image = cv2.cvtColor(ori_image,cv2.COLOR_RGB2HSV_FULL)
    h=image[:,:,0].copy()
    sat=image[:,:,1].copy()
    val=image[:,:,2].copy()    
    #change to float32 to enable higher value
    sat = np.float32(sat)
    val = np.float32(val)
    #experiment negrofication
    if(tipe==0):
        sat[sat>0]+=satlevel
        val[val<255]-=vallevel
    #experiment caucasifacation
    elif(tipe==1):
        sat[sat>0]-=satlevel
        val[val<255]+=vallevel
    
    sat[sat<0]=0
    sat[sat>255]=255
    val[val>255]=255
    val[val<0]=0
    
    sat = np.uint8(sat)
    val = np.uint8(val)
    im=np.dstack((h,sat,val))
    im=cv2.cvtColor(im,cv2.COLOR_HSV2RGB_FULL)
    return im

def main(albedo,pref=1):
    #Ganti warna kulit (0=negro,1=caucasian)
    al_out=changeskincolor(albedo,pref,40,60)
    return al_out