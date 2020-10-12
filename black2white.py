import cv2
import glob
import os
import matplotlib.pyplot as plt


# Code to change the black background into white background (Must Provide Mask)
#Masking Real data
dir = "/home/cgal/testing/black_bg/sample/*.png"
mask = "/home/cgal/testing/black_bg/input_matrix/test/masking/*.png"
tar = "/home/cgal/testing/white_bg/sample/"
if not os.path.exists(tar):
    os.makedirs(tar)

for i,j in zip(sorted(glob.glob(dir)),sorted(glob.glob(mask))):
    img = cv2.imread(i)
    mask = cv2.imread(j)
    mask = cv2.bitwise_not(mask) 
    iname = i.split('/')[-1].split('.')[0]
    img[mask==255]=255
    cv2.imwrite(str(tar+iname)+".png", img)

#Masking Synthetic
# dir = "/home/cgal/testing/black_bg/target_train_synt_matrix/*.png"
# tar = "/home/cgal/testing/white_bg/target_train_synt_matrix/"
# if not os.path.exists(tar):
#     os.makedirs(tar)

# for i in sorted(glob.glob(dir)):
# 	img = cv2.imread(i)
# 	iname = i.split('/')[-1].split('.')[0]

# 	ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 10, 5, cv2.THRESH_BINARY)
# 	img[thresh == 0] = 255
# 	cv2.imwrite(str(tar+iname)+".png", img)