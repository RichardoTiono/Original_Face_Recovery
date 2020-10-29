import numpy as np
import cv2
import math


# ### Color Harmonize Part

# In[10]:

templateNames=['i', 'V', 'L', 'I', 'T', 'Y', 'X' ]
region1Arcs= [18, 94, 18, 18, 180, 94, 94 ]
region2Arcs= [0, 0, 80, 18, 0, 18, 94 ]
c1toc2 =[0, 0, 90, 180, 0, 180, 180 ]

def applytemplate(template,starting_point,height):
    t_index = templateNames.index(str(template))
    second=starting_point+region1Arcs[t_index]
    new_hist=np.zeros(360)
    if(second<=360):
        new_hist[starting_point:second]=N[starting_point:second]    
    else:
        excess=int(second%360)
        new_hist[0:excess]=N[0:excess]
        new_hist[starting_point:360]=N[starting_point:360]
    if(region2Arcs[t_index]!=0):
        third=int(starting_point+region1Arcs[t_index]/2+c1toc2[t_index]-region2Arcs[t_index]/2)
        fourth=third+region2Arcs[t_index]
        if(third<=360):
            if(fourth<=360):
                new_hist[third:fourth]=N[third:fourth]
            else:
                excess=int(fourth%360)
                new_hist[0:excess]=N[0:excess]
                new_hist[third:360]=N[third:360]
        else:
            third=int(third%360)
            fourth=third+region2Arcs[t_index]
            new_hist[third:fourth]=N[third:fourth]
    return new_hist


# In[11]:


def computedistancetoborder(template,template_point,hue_position,neigh,saturation,sat_neigh):
    t_index=templateNames.index(template)
    #Region 1 of template
    border1 = template_point
    border2 = (template_point+region1Arcs[t_index])%360
    hue_position = hue_position
    region = 1
    orientation=1 #-1=counterclockwise 1 = clockwise
    tempdistance=999
    tempenergy=99999999999 #Default value if there are no region 2
    ori2=1
    #Formula for Energy
    compare = np.where(neigh==hue_position,0,1)
    maks=np.maximum(saturation,sat_neigh)
    h=1/np.absolute(np.subtract(hue_position,neigh))
    h=np.where(np.isinf(h), 0, h)
    E2=np.sum(compare*maks*h)
    #cek di dalam region 1 atau tidak
    if(border1<=hue_position) and (hue_position<=border2):
        distance = 0
        energy=0
    #kalau border 1 sebelum 360 dan border 2 nya setelah 360/0
    elif(border1>border2) and ((border1<=hue_position) or (hue_position<=border2)):
        distance = 0
        energy=0
    else:
        #ngecek ke terdekat
        d1 = abs(hue_position-border1)
        d2 = abs(hue_position-border2)
        #ngecek ke sisi yg lainnya 
        d3 = abs(360-d1)
        d4 = abs(360-d2)
        min1=min(d1,d2)
        min2=min(d3,d4)
        if(min1<min2):
            distance=min1
            if(hue_position>border1):
                ori1=-1
            elif(hue_position<border1):
                ori1=1
        else :
            distance=min2
            if(hue_position<border1):
                ori1=1
            elif(hue_position>border1):
                ori1=-1
        #distance=min(d1,d2,d3,d4)
        orientation=ori1
        E1=distance*saturation
        energy=E1+E2
    #Region 2 if available
    if(region2Arcs[t_index]!=0):
        border1=(border1+(region1Arcs[t_index]/2)+c1toc2[t_index]-(region2Arcs[t_index]/2))%360
        border2=(border1+region2Arcs[t_index])%360
        if(border1<=hue_position) and (hue_position<=border2):
            tempdistance = 0
            tempenergy=0
        elif(border1>border2) and ((border1<=hue_position) or (hue_position<=border2)):
            tempdistance = 0
            tempenergy=0
        else:
            #ngecek ke terdekat
            d1 = abs(hue_position-border1)
            d2 = abs(hue_position-border2)
            #ngecek ke sisi yg lainnya
            d3 = abs(360-d1)
            d4 = abs(360-d2)        
            min1=min(d1,d2)
            min2=min(d3,d4)
            if(min1<min2):
                tempdistance=min1
                if(hue_position>border1):
                    ori2=-1
                elif(hue_position<border1):
                    ori2=1
            else :
                tempdistance=min2
                if(hue_position<border1):
                    ori2=1
                elif(hue_position>border1):
                    ori2=-1
            #tempdistance=min(d1,d2,d3,d4)
            E1=tempdistance*saturation
            tempenergy=E1+E2
    #Check between region 1 or region 2 is closest
    if(energy>tempenergy):
        distance=tempdistance
        region = 2
        orientation=ori2
    return region,distance,orientation


# In[12]:


def autotemplate(ori_image):
    image = cv2.cvtColor(ori_image,cv2.COLOR_RGB2HSV_FULL)
    row,col=image[:,:,0].shape
    ori_hues=(image[:,:,0]/255*360).reshape(row*col) 
    saturation = image[:,:,1].reshape(row*col) 
    temp_hues = ori_hues.astype(int)
    compatibility_matrix = np.zeros((360,len(templateNames)))
    for arc in range(360):
        for t in range(len(templateNames)):
            distance_matrix = temp_hues.copy()
            for i in range(360):
                _,distance_value,_=computedistancetoborder(templateNames[t],arc,i,np.zeros(8),0,np.zeros(8))
                np.place(distance_matrix,temp_hues==i,distance_value)
            temp = np.sum(np.multiply(distance_matrix,saturation))
            compatibility_matrix[arc,t]=temp
    best=np.min(compatibility_matrix)
    result=np.where(compatibility_matrix==best)
    arc,template = result[0],result[1]
    return arc,template


# In[13]:


def computedistancetocenter(template,template_point,hue_position,region):
    t_index=templateNames.index(template)
    center = (template_point+region1Arcs[t_index]/2)%360
    tempdistance=999 #Default value if there are no region 2
    #center of region1
    if(region==1):
        #ngecek ke kiri
        d1 = abs(hue_position-center)
        #ngecek ke kanan
        d2 = abs(360-d1)
        distance=min(d1,d2)
    #Region 2 if available
    elif(region==2):
        center=(center+c1toc2[t_index])%360
        #ngecek ke kiri
        d1 = abs(hue_position-center)
        #ngecek ke kanan
        d2 = abs(360-d1)
        distance=min(d1,d2)
    #Check between region 1 or region 2 is closest
    return distance


# In[14]:


def fittoregion(template,template_point,hue_point,neigh,saturation,sat_neigh):
    t_index=templateNames.index(template)
    region,_,orientation= computedistancetoborder(template,template_point,hue_point,neigh,saturation,sat_neigh)
    distance= computedistancetocenter(template,template_point,hue_point,region)
    if(region==1):
        center = (template_point+region1Arcs[t_index]/2)%360
        w = region1Arcs[t_index]
    else:
        center = (template_point+region1Arcs[t_index]/2+c1toc2[t_index])%360
        w = region2Arcs[t_index]
    new_value=round(center+orientation*(w/2)*(1-(math.exp(-0.5*((distance/(w/2))**2)))))%360
    return new_value


# In[15]:


def recalculatehist(im,template,template_point):
    image=im.copy()
    hues=(image[:,:,0]/255*360).astype(int)
    saturation = image[:,:,1].astype(int)
    new_hues=hues.copy()
    hues = np.pad(hues, pad_width=1, mode='constant', constant_values=500)
    saturation = np.pad(saturation, pad_width=1, mode='constant', constant_values=500)
    row,col=hues.shape
    #convert->area yang harus dipindah
    #new_mapping = (value lama, value baru)
    for i in range(1,row-1):
        for j in  range(1,col-1):
            hue_neigh=hues[i-1:i+2,j-1:j+2].flatten()
            sat_neigh=saturation[i-1:i+2,j-1:j+2].flatten()
            #delete middle matrix
            hue_neigh=np.delete(hue_neigh,4)
            sat_neigh=np.delete(sat_neigh,4)
            #delete padding 
            hue_neigh = hue_neigh[~np.in1d(hue_neigh,500)]
            sat_neigh = sat_neigh[~np.in1d(sat_neigh,500)]
            new_value=fittoregion(template,template_point,hues[i,j],hue_neigh,saturation[i,j],sat_neigh)
            new_hues[i-1,j-1]=new_value
    return new_hues


# In[16]:


def color_harmonize(img,tipe,template,degree = 0):
    ori_image=img.copy()
    image = cv2.cvtColor(ori_image,cv2.COLOR_RGB2HSV_FULL)
    hues=np.round(image[:,:,0]/255*360)
    saturation = image[:,:,1]
    new_hues=hues.copy()
    hues = np.pad(hues, pad_width=1, mode='constant', constant_values=0)
    saturation = np.pad(saturation, pad_width=1, mode='constant', constant_values=0)
    row,col=hues.shape
    if(tipe=='manual'):
        template = template
        template_point= int(degree)
        for i in range (1,row-1):
            for j in range (1,col-1):
                neighbour=hues[i-1:i+2,j-1:j+2].flatten()
                sat_neigh=saturation[i-1:i+2,j-1:j+2].flatten()
                #delete middle matrix
                neighbour=np.delete(neighbour,4)
                sat=sat_neigh[4]
                sat_neigh=np.delete(sat_neigh,4)
                new_value=fittoregion(template,template_point,hues[i,j],neighbour,sat,sat_neigh)
                new_hues[i-1,j-1]=new_value
        new_hues=np.round(new_hues/360*255)
        image[:,:,0]=new_hues
        new_image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB_FULL)
    elif(tipe=='Auto'):
        x,template = autotemplate(img)
        print('best template: ',templateNames[template[-1]],  ' best Template Position:',x[-1],)
        new_hues = recalculatehist(image,templateNames[template[-1]],int(x[-1]))
        image[:,:,0]=np.round(new_hues/360*255)
        new_image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB_FULL)
    return new_image


def main(albedo, template ,degree):
    if template =='Auto':
        al_out=color_harmonize(img = albedo,tipe = 'Auto', template = None,degree = 0)  
    else :
        al_out=color_harmonize(img = albedo,tipe = 'manual', template= template ,degree = degree)  
    return al_out