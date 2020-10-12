import cv2
import ntpath
import glob
import matplotlib.pyplot as plt
import os


# Code to combine mask and image 


# #synthethic
# path = './dataset/Syn_data/train/*/'
# list_mask=[f for f in sorted(glob.glob(path + '*_mask_*'))]
# list_im=[g for g in sorted(glob.glob(path + '*_face_*'))]

# total = len(list_im)
# for img in range(total):
# 	name=ntpath.basename(list_im[img])
# 	name=name.replace("face","combine")
# 	dirname=os.path.dirname(list_im[img])
# 	im = cv2.imread(list_im[img])
# 	#im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# 	im=cv2.resize(im, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
# 	mask = cv2.imread(list_mask[img])
# 	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# 	combine = cv2.bitwise_and(im,im,mask = mask)
# 	cv2.imwrite(dirname+'/'+name,combine)

# #HR
# path = '/home/cgal/testimageSFS/CelebAMask-HQ'
# list_mask=[f for f in sorted(glob.glob(path + '/HRmask/*/*_skin*'))]
# list_im=[g for g in sorted(glob.glob(path + '/HRceleba/*'), key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))]
# directory = path + '/HRcombine/'
# os.system('mkdir -p {}'.format(directory))

# total = len(list_im)
# for img in range(total):
# 	im = cv2.imread(list_im[img])
# 	im=cv2.resize(im, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
# 	mask = cv2.imread(list_mask[img])
# 	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# 	combine = cv2.bitwise_and(im,im,mask = mask)
# 	cv2.imwrite(directory + str(img)+'.png',combine)

path = '/home/cgal/testimageSFS/CelebAMask-HQ'
list_mask=[f for f in sorted(glob.glob(path + '/HRmask/*/*_skin*'))]
list_im=[g for g in sorted(glob.glob(path + '/HRceleba/*'), key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))]
directory = path + '/HRcombine/'
os.system('mkdir -p {}'.format(directory))

total = len(list_im)
for img in range(total):
	im = cv2.imread(list_im[img])
	im=cv2.resize(im, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
	mask = cv2.imread(list_mask[img])
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	combine = cv2.bitwise_and(im,im,mask = mask)
	cv2.imwrite(directory + str(img)+'.png',combine)