import torch
from models import *
from torchsummary import summary
from utils import *
import argparse



def main():
	parser = argparse.ArgumentParser(description='check model architecture')
	parser.add_argument('--model', type=int, default=0,
	                        help='0 = skipnet ')
	parser.add_argument('--no_cuda', action='store_true', default=False,
	                        help='disables CUDA training')

	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	model_selection = args.model

	if(model_selection==0):

		skipnet_model      = SkipNet()
		if use_cuda:
		        skipnet_model = skipnet_model.cuda()

		skipnet_model.load_state_dict(torch.load('./results/Synthetic_Train/checkpoints/skipnet_model.pkl'))
		print(skipnet_model)
	elif(model_selection==1):
		sfs_net_model      = SfsNetPipeline()
		pretrained_model_dict='./pretrained/net_epoch_r5_5.pth'
		sfs_net_model.apply(weights_init)
		sfs_net_pretrained_dict = torch.load(pretrained_model_dict)
		sfs_net_state_dict = sfs_net_model.state_dict()
		load_model_from_pretrained(sfs_net_pretrained_dict, sfs_net_state_dict)
		sfs_net_model.load_state_dict(sfs_net_state_dict)
		sfs_net_model.fix_weights()
		print(sfs_net_model)
if __name__ == '__main__':
    main()