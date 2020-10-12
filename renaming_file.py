import os
import glob

target_image = "/home/cgal/testing/black_bg/target_matrix/test"
generated_image = "/home/cgal/testing/black_bg/rev_faces_cyclegan"

for img,target in zip(sorted(glob.glob(generated_image + '/*.png')), sorted(glob.glob(target_image + '/*.png'))):

	iname = target.split('/')[-1].split('.')[0]
	dir = os.path.split(img)[0] 
	os.rename(img, dir+"/"+iname+".png")