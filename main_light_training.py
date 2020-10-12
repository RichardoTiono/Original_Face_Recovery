import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
from torchsummary import summary

from data_loading import *
from utils import *
from train import *
from models import *
import json

#Train & Testing Original Face Recovery Network

def main():
    ON_SERVER = False

    parser = argparse.ArgumentParser(description='Skin Recovery Network')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--wt_decay', type=float, default=0.0005, metavar='W',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training (FALSE = USING CUDA)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--read_first', type=int, default=-1,
                        help='read first n rows (default: -1)')
    parser.add_argument('--details', type=str, default="Light Extraction Training",
                        help='Explaination of the run')
    parser.add_argument('--sample_data', type=str, default='/home/cgal/testing/black_bg/matrix_my',#'./dataset/input_train_real_matrix',
                        help='Sample Dataset path - Data for testing/inferring with your pretrained network')
    parser.add_argument('--rgb_syn_data', type=str, default='./dataset/rgb_synt_central_lighting',
                        help='RGB Synt Dataset path')
    parser.add_argument('--rgb_real_data', type=str, default='./dataset/rgb_real_masking',
                    help='RGB Real Dataset path')
    parser.add_argument('--log_dir', type=str, default='./results/',
                        help='Log Path')
    parser.add_argument('--load_model', type=str, default='./pretrained_models/',
                        help='load model from')
    parser.add_argument('--mode', type=str, default='metrics',
                        help='Decide doing Train-rgb (Training Network) or metrics(test using metrics) or Infer (Infer pretrained network) '
                        'or Visualization (visuale each convolution output in grayscale)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # initialization
    sample_data = args.sample_data
    rgb_synt_data = args.rgb_syn_data
    rgb_real_data = args.rgb_real_data
    batch_size = args.batch_size
    lr         = args.lr
    wt_decay   = args.wt_decay
    log_dir    = args.log_dir
    epochs     = args.epochs
    model_dir  = args.load_model
    read_first = args.read_first
    mode = args.mode

    
    if read_first == -1:
        read_first = None

    #Initialize models
    generator = Generator()
    discriminator = Discriminator()
    perceptual_loss = VGGPerceptualLoss()
    result_dir = args.log_dir+'Light_Training/'

    if (mode!='metrics'):
        #Configure Cuda and Data parallel for multiple GPU
        if use_cuda:
            print("Number of GPU used :", torch.cuda.device_count(), "GPUs!")
            generator = nn.DataParallel(generator)
            generator = generator.cuda()
            discriminator = nn.DataParallel(discriminator)
            discriminator = discriminator.cuda()
            perceptual_loss = perceptual_loss.cuda()

        #Load pretrained network 
        if model_dir is not None:
            print("using pretrained model")
            checkpoint = torch.load(model_dir + 'light_gan10.pkl')
            generator.apply(weights_init)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            g_optim_state= checkpoint['g_optimizer_dict']
            last_epoch = checkpoint['epoch']

        else :        
            print("creating new model")
            d_optim_state= None
            g_optim_state= None
            last_epoch= None
            generator.apply(weights_init)      
            discriminator.apply(weights_init) 

    #Choose Mode
    if(mode=='Train-rgb') : 
        print("Training RGB")
        print("Initialize weight")        
        os.system('mkdir -p {}'.format(result_dir))
        #Create log for recording training details
        with open(result_dir+'details.txt', 'w') as f:
            f.write(args.details)
        #Create log for recording validation details
        with open(result_dir+'val_details.txt', 'w') as f:
            f.write(args.details)
        #create log for network setting and architecture details
        with open(result_dir+'network_details.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print(generator)
        #Write generator and discriminator architecture
        with open(result_dir+'network_details.txt', 'a') as f:
            f.write('\n'+str(generator)) 
        with open(result_dir+'network_details.txt', 'a') as f:
            f.write('\n'+str(discriminator))   

        #begin Training
        train_new_style(generator=generator,discriminator=discriminator, rgb_syn_data = rgb_synt_data,rgb_real_data = rgb_real_data, read_first=read_first,\
                batch_size=batch_size, num_epochs=epochs, log_path=result_dir, use_cuda=use_cuda, \
                lr=lr, wt_decay=wt_decay,last_g_state=g_optim_state, last_d_state=d_optim_state, last_epoch = last_epoch, perceptual_loss = perceptual_loss)

        print("Training Complete")

    elif(mode=='metrics') : 
        print("Check metrics result")
        target_image = "/home/cgal/testing/white_bg/target_matrix"
        generated_image = "/home/cgal/testing/white_bg/relit"
        #generated_image = "/home/cgal/SfSNet/result"

        sample_dataset,_ = get_light_dataset(gen_dir=generated_image,target_dir=target_image, validation_split=0)
        #test_dataset, _ = get_celeba_dataset(read_from_csv=celeba_test_csv, read_first=read_first, validation_split=0)
        sample_dl = DataLoader(sample_dataset, batch_size=1, shuffle=True)
        out_sample_images_dir = './results/ssim2/'
        #print(out_celeba_images_dir)
        os.system('mkdir -p {}'.format(out_sample_images_dir))

        # Dump normal, albedo, shading, face and sh for celeba dataset
        Calculate_metrics( sample_dl, use_cuda = use_cuda, out_folder = out_sample_images_dir)

    elif(mode=='Infer') : 
        generator.eval()
        sample_dataset = get_sample_dataset(dir=sample_data,read_first=read_first, validation_split=0)
        #test_dataset, _ = get_celeba_dataset(read_from_csv=celeba_test_csv, read_first=read_first, validation_split=0)
        sample_dl  = DataLoader(sample_dataset, batch_size=1, shuffle=True)
        out_sample_images_dir = './results/sample/'
        #print(out_celeba_images_dir)
        os.system('mkdir -p {}'.format(out_sample_images_dir))

        # Dump normal, albedo, shading, face and sh for celeba dataset

        generate_light_extractor_sample(generator, sample_dl, train_epoch_num = 0,
                                use_cuda = use_cuda, out_folder = out_sample_images_dir)

    elif(mode=='Visualization'):
        generator.eval()
        sample_dataset = get_sample_dataset(dir=sample_data,read_first=read_first, validation_split=0)
        #test_dataset, _ = get_celeba_dataset(read_from_csv=celeba_test_csv, read_first=read_first, validation_split=0)
        sample_dl  = DataLoader(sample_dataset, batch_size=1, shuffle=True)
        out_sample_images_dir = './results/visualization/'
        #print(out_celeba_images_dir)
        os.system('mkdir -p {}'.format(out_sample_images_dir))

        # Dump normal, albedo, shading, face and sh for celeba dataset

        visualize_activation(generator, sample_dl, output_folder = out_sample_images_dir,use_cuda=use_cuda)



if __name__ == '__main__':
    main()