#
# Experiment Entry point
# 1. Trains model on Syn Data
# 2. Generates CelebA Data
# 3. Trains on Syn + CelebA Data
#

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse

from data_loading import *
from utils import *
from shading import *
from train import *
from models import *

def main():
    ON_SERVER = False

    parser = argparse.ArgumentParser(description='SfSNet - Residual')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--wt_decay', type=float, default=0.0005, metavar='W',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--read_first', type=int, default=10,
                        help='read first n rows (default: -1)')
    parser.add_argument('--details', type=str, default=None,
                        help='Explaination of the run')
    parser.add_argument('--load_pretrained_model', type=str, default='./pretrained/net_epoch_r5_5.pth',
                        help='Pretrained model path')
    parser.add_argument('--syn_data', type=str, default='./dataset/Syn_data',
                    help='Synthetic Dataset path')
    parser.add_argument('--celeba_data', type=str, default='./dataset/HRcombine_synt128',
                    help='CelebA Dataset path')
    parser.add_argument('--sample_data', type=str, default='./dataset/testimagecolor/masking/',
                    help='Sample Dataset path')
    parser.add_argument('--rgb_syn_data', type=str, default='./dataset/rgb_synt_new',
                        help='RGB Synt Dataset path')
    parser.add_argument('--rgb_real_data', type=str, default='./dataset/rgb_real_masking',
                    help='RGB Real Dataset path')
    parser.add_argument('--log_dir', type=str, default='./results/',
                    help='Log Path')
    parser.add_argument('--load_model', type=str, default='./results/',
                        help='load model from')
    parser.add_argument('--mode', type=str, default='Pretrained',
                        help='Decide doing Train-sfs or Testing or Create')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # initialization
    syn_data = args.syn_data
    celeba_data = args.celeba_data
    rgb_synt_data = args.rgb_syn_data
    rgb_real_data = args.rgb_real_data
    batch_size = args.batch_size
    sample_data = args.sample_data
    lr         = args.lr
    wt_decay   = args.wt_decay
    log_dir    = args.log_dir
    epochs     = args.epochs
    model_dir  = args.load_model
    read_first = args.read_first
    mode = args.mode
    pretrained_model_dict = args.load_pretrained_model

    
    if read_first == -1:
        read_first = None

    # Debugging and check working
    # syn_train_csv = syn_data + '/train.csv'
    # train_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data+'train/', read_from_csv=syn_train_csv, read_celeba_csv=None, read_first=read_first, validation_split=5)
    # train_dl  = DataLoader(train_dataset, batch_size=10, shuffle=False)
    # validate_shading_method(train_dl)
    # return 

    # Initialize models
    sfs_net_model      = SfsNetPipeline()
    #sfs_net_model      = SfSNet()
    perceptual_loss = VGGPerceptualLoss()

    if use_cuda:
        print("use Cuda")
        sfs_net_model = sfs_net_model.cuda()
        sfs_net_model = nn.DataParallel(sfs_net_model)
        perceptual_loss = perceptual_loss.cuda()

    if model_dir is not None:
        sfs_net_model.apply(weights_init)
        print("Loading pretrained model")
        sfs_net_model.load_state_dict(torch.load(model_dir + 'sfs_net_model_9.pkl')['model_state_dict'],strict = False)
        print("Fixing model weight")
        sfs_net_model.module.fix_weights()
    else:
        sfs_net_model.apply(weights_init)


    if model_dir is not None:
        if mode =='Create':
            sfs_net_model.eval()
            print("Getting dataset")
            train_dataset,test_dataset = get_celeba_dataset(dir=sample_data ,read_from_csv=None, read_first=read_first, validation_split=2)

            print("loading to dataloader")
            celeba_sample_train  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            celeba_sample_test  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            out_folder = './results/HRsfscaffe/'
            out_train_celeba_images_dir = out_folder + 'train/'
            out_test_celeba_images_dir = out_folder + 'test/'

        #print(out_celeba_images_dir)
            os.system('mkdir -p {}'.format(out_train_celeba_images_dir))
            os.system('mkdir -p {}'.format(out_test_celeba_images_dir))

            print("start generating training data")
            generate_celeba_synthesize(sfs_net_model, celeba_sample_train, train_epoch_num=epochs, use_cuda=use_cuda,
                                                                out_folder=out_train_celeba_images_dir)
            print("start generating test data")
            generate_celeba_synthesize(sfs_net_model, celeba_sample_test, train_epoch_num=epochs, use_cuda=use_cuda,
                                                                out_folder=out_test_celeba_images_dir)
            print("finish generating all data")
        elif mode =="Testing":
            sfs_net_model.eval()
            print("Getting dataset")
            sample_dataset,_ = get_celeba_dataset(dir=sample_data ,read_from_csv=None, read_first=read_first, validation_split=0)

            print("loading to dataloader")
            sample_loader  = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True)

            out_folder = './results/sample_result/'

        #print(out_celeba_images_dir)
            os.system('mkdir -p {}'.format(out_folder))

            print("start generating sample data")
            generate_celeba_synthesize(sfs_net_model, sample_loader, train_epoch_num=epochs, use_cuda=use_cuda,
                                                                out_folder=out_folder)
            print("Finish testing all data")
        else :
            #Train using pretrain for light extraction
            print("Start transfer learning for light correction")
            train_withPretrain(sfs_net_model, syn_data, celeba_data=celeba_data, rgb_syn_data = rgb_synt_data,rgb_real_data = rgb_real_data, read_first=read_first,\
                batch_size=batch_size, num_epochs=epochs, log_path=log_dir+'Mix_Training/', use_cuda=use_cuda, \
                lr=lr, wt_decay=wt_decay, perceptual_loss = perceptual_loss)
    else :
    # 1. Train on both Synthetic and Real (Celeba) dataset from beginning
        train(sfs_net_model, syn_data, celeba_data=celeba_data, read_first=read_first,\
                batch_size=batch_size, num_epochs=epochs, log_path=log_dir+'Mix_Training/', use_cuda=use_cuda, \
                lr=lr, wt_decay=wt_decay)
if __name__ == '__main__':
    main()