from data_loading import *
import os, glob

#Code to generate CSV file for dataset

#Generate synthetic
# dataset_path = './dataset/Syn_data/'
# generate_synthesize_data_csv(dataset_path + 'train/*/', dataset_path + '/train.csv')
# generate_synthesize_data_csv(dataset_path + 'test/*/', dataset_path + '/test.csv')

# # #Generate celeba
# celeba_path = './dataset/HRcombine_synt128/'
# generate_celeba_synthesize_data_csv(celeba_path + 'train/', celeba_path + '/train.csv')
# generate_celeba_synthesize_data_csv(celeba_path + 'test/', celeba_path + '/test.csv')

#Generate RGB synthetic
rgb_path = './dataset/rgb_synt_central_lighting/'

generate_rgb_csv(dir=rgb_path+'images',train_subject=8, save_location=rgb_path)

# #generate RGB real
rgb_real = './dataset/rgb_real_masking/'
generate_rgb_real_csv(rgb_real ,train_subject=20, save_location =rgb_real)