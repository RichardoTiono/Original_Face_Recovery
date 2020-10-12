import matplotlib.pyplot as plt
import torchvision
import cv2
from PIL import Image
from torch.nn import *
import torch


def gradient_penalty(y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

def applyMask(input_img, mask):
    if mask is None:
        return input_img
    return input_img * mask

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def get_normal_in_range(normal):
    new_normal = normal * 128 + 128
    new_normal = new_normal.clamp(0, 255) / 255
    return new_normal

def get_image_grid(pic, denormalize=False, mask=None):
    if denormalize:
        pic = denorm(pic)
    
    if mask is not None:
        pic = pic * mask

    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return ndarr

def save_image(pic, denormalize=False, path=None, mask=None):
    ndarr = get_image_grid(pic, denormalize=denormalize, mask=mask)  
    if path == None:
        plt.imshow(ndarr)
        plt.show()
    else:
        im = Image.fromarray(ndarr)
        im.save(path)

def wandb_log_images(wandb, img, mask, caption, step, log_name, path=None, denormalize=False):
    ndarr = get_image_grid(img, denormalize=denormalize, mask=mask)

    # save image if path is provided
    if path is not None:
        im = Image.fromarray(ndarr)
        im.save(path)

    wimg = wandb.Image(ndarr, caption=caption)
    wandb.log({log_name: wimg})

def weights_init(m):
    if isinstance(m, Conv2d) or isinstance(m, Conv1d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            init.constant_(m.bias, 0)
    elif isinstance(m, Linear):
        init.normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
