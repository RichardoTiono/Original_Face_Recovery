import shutil, os
import pandas as pd

#Code to move dataset from 1 folder to another folder

face = []
target = []
names = []
#read_rgb_real_csv = "./dataset/rgb_synt_central_lighting/rgb_train.csv"
read_rgb_real_csv = "./dataset/rgb_real_masking/rgb_real_train.csv"
if read_rgb_real_csv is not None:
    df = pd.read_csv(read_rgb_real_csv)
    face   += list(df['face'])
    target     += list(df['target'])
    names += list(df['name'])

dir = '/home/cgal/testing/black_bg/input_train_real_matrix/'
tar = '/home/cgal/testing/black_bg/target_train_real_matrix/'
if not os.path.exists(dir):
    os.makedirs(dir)
if not os.path.exists(tar):
    os.makedirs(tar)

data_len = len(face)
for f in range(0,data_len,21):
	#Train
    # shutil.copy(face[f], '/home/cgal/cpixgan/datasets/datasets/Pixfaces/A/train/'+str(names[f])+'.png')
    # shutil.copy(target[f], '/home/cgal/cpixgan/datasets/datasets/Pixfaces/B/train/'+str(names[f])+'.png')

    shutil.copy(face[f], dir+str(names[f])+'.png')
    shutil.copy(target[f], tar+str(names[f])+'.png')