import torch
import torch.nn as nn
import numpy as np
import os

from models import *
from utils import *
from data_loading import *
import cv2
from metrics import compute_psnr,compute_ssim
import pytorch_msssim

## TODOS:
## 1. Dump SH in file
## 
## 
## Notes:
## 1. SH is not normalized
## 2. Face is normalized and denormalized - shall we not normalize in the first place?


# Enable WANDB Logging
WANDB_ENABLE = False

def predict_celeba(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'CelebA_Val', dump_all_images = False):
 
    # debugging flag to dump image
    fix_bix_dump = 0
    recon_loss  = nn.L1Loss() 

    if use_cuda:
        recon_loss  = recon_loss.cuda()

    tloss = 0 # Total loss
    rloss = 0 # Reconstruction loss
    for bix, data in enumerate(dl):
        face = data
        if use_cuda:
            face   = face.cuda()
        
        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(face)
        # coba
        # file_name = out_folder 
        # save_image(predicted_normal, path = file_name+'_normal.png')
        # save_image(predicted_albedo, path = file_name+'_albedo.png')
        # save_image(predicted_shading, path = file_name+'_shading.png')
        # save_image(predicted_face, path = file_name+'_recon.png')
        if bix == fix_bix_dump or dump_all_images:
            # save predictions in log folder
            file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(bix)
            # log images
            predicted_normal = get_normal_in_range(predicted_normal)
            save_image(predicted_normal, path = file_name+'_normal.png')
            save_image(predicted_albedo, path = file_name+'_albedo.png')
            save_image(predicted_shading, path = file_name+'_shading.png')
            save_image(predicted_face, path = file_name+'_recon.png')
            save_image(face, path = file_name+'_face.png')
            # TODO:
            # Dump SH as CSV or TXT file
        
        # Loss computation
        # Reconstruction loss
        total_loss  = recon_loss(predicted_face, face)

        # Logging for display and debugging purposes
        tloss += total_loss.item()
    
    len_dl = len(dl)
    if(wandb is not None):
        wandb.log({suffix+' Total loss': tloss/len_dl}, step=train_epoch_num)
            

    # return average loss over dataset
    return tloss / len_dl

def predict_synthetic(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'Val'):
 
    # debugging flag to dump image
    #sfs_net_model.eval()

    fix_bix_dump = 0
    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss     = nn.MSELoss()
    recon_loss  = nn.L1Loss() 
    corrector_loss  = nn.L1Loss() 

    lamda_recon  = 0.5
    lamda_albedo = 0.5
    lamda_normal = 0.5
    lamda_sh     = 0.1
    lamda_cor     = 0.5

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()
        corrector_loss = corrector_loss.cuda()

    tloss = 0 # Total loss
    nloss = 0 # Normal loss
    aloss = 0 # Albedo loss
    shloss = 0 # SH loss
    rloss = 0 # Reconstruction loss
    closs = 0


    for bix, data in enumerate(dl):
        albedo, normal, sh, face = data
        if use_cuda:
            albedo = albedo.cuda()
            normal = normal.cuda()
            sh     = sh.cuda()
            face   = face.cuda()
        
        # Apply Mask on input image
        # face = applyMask(face, mask)
        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(face)

        if fix_bix_dump == bix :
            # save predictions in log folder
            save_p_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)
            # log images
            for i in range(face.size()[0]):
                file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(i) + '_' + str(bix)+'.png'

                tensor = torch.cat([predicted_albedo.unsqueeze(0)[:,i,:,:], save_p_normal.unsqueeze(0)[:,i,:,:],predicted_shading.unsqueeze(0)[:,i,:,:], \
                    predicted_face.unsqueeze(0)[:,i,:,:],face.unsqueeze(0)[:,i,:,:]], dim=0)
                utils.save_image(tensor.cpu(), file_name , nrow=5)
             
        
        # Loss computation
        # Normal loss
        current_normal_loss = normal_loss(predicted_normal, normal)
        # Albedo loss
        current_albedo_loss = albedo_loss(predicted_albedo, albedo)
        # SH loss
        current_sh_loss     = sh_loss(predicted_sh, sh)
        # Reconstruction loss
        current_recon_loss  = recon_loss(predicted_face, face)


        total_loss = lamda_recon * current_recon_loss + lamda_normal * current_normal_loss \
                        + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss 
        # Logging for display and debugging purposes
        tloss += total_loss.item()
        nloss += current_normal_loss.item()
        aloss += current_albedo_loss.item()
        shloss += current_sh_loss.item()
        rloss += current_recon_loss.item()

    len_dl = len(dl)
            
    # return average loss over dataset
    return tloss / len_dl, nloss / len_dl, aloss / len_dl, shloss / len_dl, rloss / len_dl

def predict_sfsnet(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'Val'):
 
    # debugging flag to dump image
    #sfs_net_model.eval()

    fix_bix_dump = 0
    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss     = nn.MSELoss()
    recon_loss  = nn.L1Loss() 

    lamda_recon  = 0.5
    lamda_albedo = 0.5
    lamda_normal = 0.5
    lamda_sh     = 0.1

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()

    tloss = 0 # Total loss
    nloss = 0 # Normal loss
    aloss = 0 # Albedo loss
    shloss = 0 # SH loss
    rloss = 0 # Reconstruction loss


    for bix, data in enumerate(dl):
        albedo, normal, sh, face ,faceHR= data
        if use_cuda:
            albedo = albedo.cuda()
            normal = normal.cuda()
            sh     = sh.cuda()
            face   = face.cuda()
            faceHR   = faceHR.cuda()
        
        # Apply Mask on input image
        # face = applyMask(face, mask)
        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(faceHR)
        #if fix_bix_dump == bix :
            # save predictions in log folder
        save_p_normal = get_normal_in_range(predicted_normal)
        save_gt_normal = get_normal_in_range(normal)
        # log images
        for i in range(face.size()[0]):
            file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(i) + '_' + str(bix)+'.png'

            tensor = torch.cat([predicted_albedo.unsqueeze(0)[:,i,:,:], save_p_normal.unsqueeze(0)[:,i,:,:],predicted_shading.unsqueeze(0)[:,i,:,:], \
                predicted_face.unsqueeze(0)[:,i,:,:],face.unsqueeze(0)[:,i,:,:]], dim=0)
            utils.save_image(tensor.cpu(), file_name , nrow=5)
             
        
        # Loss computation
        # Normal loss
        current_normal_loss = normal_loss(predicted_normal, normal)
        # Albedo loss
        current_albedo_loss = albedo_loss(predicted_albedo, albedo)
        # SH loss
        current_sh_loss     = sh_loss(predicted_sh, sh)
        # Reconstruction loss
        current_recon_loss  = recon_loss(predicted_face, face)


        total_loss = lamda_recon * current_recon_loss + lamda_normal * current_normal_loss \
                        + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss
        # Logging for display and debugging purposes
        tloss += total_loss.item()
        nloss += current_normal_loss.item()
        aloss += current_albedo_loss.item()
        shloss += current_sh_loss.item()
        rloss += current_recon_loss.item()

    len_dl = len(dl)
            
    # return average loss over dataset
    return tloss / len_dl, nloss / len_dl, aloss / len_dl, shloss / len_dl, rloss / len_dl 

def predict_corrector(model, dl, perceptual_loss=None, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'Val'):
 
    # debugging flag to dump image
    #sfs_net_model.eval()

    fix_bix_dump = 0

    albedo_corrector_loss = nn.L1Loss()
    shading_corrector_loss = nn.L1Loss()
    recon_corrector_loss = nn.L1Loss()

    alcorloss = 0 
    shcorloss = 0
    recorloss = 0

    for bix, data in enumerate(dl):
        attr, face,target = data
        if use_cuda:
            target   = target.cuda()
            face   = face.cuda()
            attr = attr.cuda()
        
        _, target_albedo, _, target_shading, target_recon, _, _,_= model(target)
        _, _, _, _, _, albedo_correct, shading_correct,recon_correct= model(face)


        if fix_bix_dump == bix :
            # save predictions in log folder
            # save_p_normal = get_normal_in_range(predicted_normal)
            # save_gt_normal = get_normal_in_range(normal)
            # log images
            for i in range(face.size()[0]):
                file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(i) + '_' + str(bix)
                tensor = torch.cat([face.unsqueeze(0)[:,i,:,:],albedo_correct.unsqueeze(0)[:,i,:,:],target_albedo.unsqueeze(0)[:,i,:,:],\
                    shading_correct.unsqueeze(0)[:,i,:,:],target_shading.unsqueeze(0)[:,i,:,:],recon_correct.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:]], dim=0)
                utils.save_image(tensor.cpu(), file_name+'.png' , nrow=7)
                # save_image(recon_correct.unsqueeze(0)[:,i,:,:], path = file_name+'_recon.png')
                # save_image(target.unsqueeze(0)[:,i,:,:], path = file_name+'_target.png')
                #np.savetxt(file_name+'_cls.txt', label[i].cpu().detach().numpy(), delimiter='\t')
        
        # Loss computation

        current_albedo_cor_loss = shading_corrector_loss(albedo_correct,target_albedo) #+ perceptual_loss(albedo_correct,target_albedo)
        current_shading_cor_loss = albedo_corrector_loss(shading_correct,target_shading) + perceptual_loss(shading_correct,target_shading)
        current_recon_cor_loss = recon_corrector_loss(recon_correct,target_recon) + perceptual_loss(recon_correct,target_recon)

        alcorloss += current_albedo_cor_loss.item()
        shcorloss += current_shading_cor_loss.item()
        recorloss += current_recon_cor_loss.item()

    len_dl = len(dl)
            
    # return average loss over dataset
    return alcorloss/len_dl, shcorloss/len_dl , recorloss/len_dl

def predict_light(model, dl, perceptual_loss=None, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, iteration = None, wandb = None, suffix = 'Val'):
 
    # debugging flag to dump image

    fix_bix_dump = 0

    recon_corrector_loss = nn.L1Loss()
    perceptual_loss = perceptual_loss

    alcorloss = 0 
    shcorloss = 0
    recorloss = 0

    for bix, data in enumerate(dl):
        face,target = data
        if use_cuda:
            target   = target.cuda()
            face   = face.cuda()
        
        fake_image = model(face)
        if fix_bix_dump == bix :
            # save predictions in log folder
            # save_p_normal = get_normal_in_range(predicted_normal)
            # save_gt_normal = get_normal_in_range(normal)
            # log images
            for i in range(face.size()[0]-5,face.size()[0]):
                file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(iteration) + '_'  + str(i) + '_' + str(bix)
                tensor = torch.cat([face.unsqueeze(0)[:,i,:,:],fake_image.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:]], dim=0)
                utils.save_image(tensor.cpu(), file_name+'.png' , nrow=3)
                # save_image(fake_image.unsqueeze(0)[:,i,:,:], path = file_name+'_recon.png')
                # save_image(target.unsqueeze(0)[:,i,:,:], path = file_name+'_target.png')
                #np.savetxt(file_name+'_cls.txt', label[i].cpu().detach().numpy(), delimiter='\t')
        
        # Loss computation
        current_recon_cor_loss = 10*recon_corrector_loss(fake_image,target) + perceptual_loss(fake_image,target)

        recorloss += current_recon_cor_loss.item()

    len_dl = len(dl)
            
    # return average loss over dataset
    return recorloss/5

def predict_fixer(model,fixer, dl, perceptual_loss=None, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, iteration = None, wandb = None, suffix = 'Val'):
 
    # debugging flag to dump image

    fix_bix_dump = 0

    recon_corrector_loss = nn.L1Loss()

    alcorloss = 0 
    shcorloss = 0
    recorloss = 0

    for bix, data in enumerate(dl):
        face,target = data
        if use_cuda:
            target   = target.cuda()
            face   = face.cuda()
        
        fake = model(face)
        fake_image = fixer(fake)
        if fix_bix_dump == bix :
            # save predictions in log folder
            # save_p_normal = get_normal_in_range(predicted_normal)
            # save_gt_normal = get_normal_in_range(normal)
            # log images
            for i in range(face.size()[0]-5,face.size()[0]):
                file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(iteration) + '_'  + str(i) + '_' + str(bix)
                tensor = torch.cat([face.unsqueeze(0)[:,i,:,:],fake.unsqueeze(0)[:,i,:,:],fake_image.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:]], dim=0)
                utils.save_image(tensor.cpu(), file_name+'.png' , nrow=4)
                # save_image(fake_image.unsqueeze(0)[:,i,:,:], path = file_name+'_recon.png')
                # save_image(target.unsqueeze(0)[:,i,:,:], path = file_name+'_target.png')
                #np.savetxt(file_name+'_cls.txt', label[i].cpu().detach().numpy(), delimiter='\t')
        
        # Loss computation
        current_recon_cor_loss = recon_corrector_loss(fake_image,target) #+ perceptual_loss(fake_image,target)

        recorloss += current_recon_cor_loss.item()

    len_dl = len(dl)
            
    # return average loss over dataset
    return recorloss/5

def Calculate_PSNR(model, dl, use_cuda = False, out_folder = None):
 
    # debugging flag to dump image
    #sfs_net_model.eval()
    file = open(out_folder+'psnr_results.txt','w') 
 
    fix_bix_dump = 0
    for bix, data in enumerate(dl):
        attr, face,target = data
        if use_cuda:
            target   = target.cuda()
            face   = face.cuda()
            attr = attr.cuda()
        
        fake_image = model(face)

        for i in range(face.size()[0]):
            file_name = out_folder + str(bix) + '_' + str(i)
            tensor = torch.cat([face.unsqueeze(0)[:,i,:,:],fake_image.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:]], dim=0)
            utils.save_image(tensor.cpu(), file_name+'.png' , nrow=3)
            psnr = compute_psnr(fake_image.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:])
            file.write(str(bix) + " PSNR results :  "+str(psnr.item())+'\n')
            #torch.save(psnr, file_name)
        


    file.close()

def Calculate_metrics(dl, use_cuda = False, out_folder = None):
 
    # debugging flag to dump image
    #sfs_net_model.eval()
    total_ssim = 0
    total_psnr = 0
    total_ms = 0
    fix_bix_dump = 0
    for bix, data in enumerate(dl):
        face,target = data

        for i in range(face.size()[0]):
            file_name = out_folder + str(bix) + '_' + str(i)
            tensor = torch.cat([face.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:]], dim=0)
            utils.save_image(tensor.cpu(), file_name+'.png' , nrow=2)
            ssim = compute_ssim(face.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:])
            psnr = compute_psnr(face.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:])
            mssim= pytorch_msssim.msssim(face.unsqueeze(0)[:,i,:,:],target.unsqueeze(0)[:,i,:,:])
            # with open(out_folder+'ssim_results.txt', 'a') as f:
            #     f.write('\n'+str(bix)+". ssim score : "+str(ssim.item())) 
            total_ssim += ssim.item()
            total_psnr += psnr.item()
            total_ms += mssim.item()
            print(file_name +": " + str(ssim.item()) +"+"+ str(psnr.item()) +"+"+ str(mssim.item()))

    print("Average SSIM : " +str(total_ssim/len(dl)))
    print("Average PSNR : " +str(total_psnr/len(dl)))
    #print("Average MS-SSIM : " +str(total_ms/len(dl)))
    # with open(out_folder+'ssim_results.txt', 'a') as f:

    #     f.write('\n'+"average score : "+str(total/len(dl))) 


def train_synthetic(sfs_net_model, syn_data, celeba_data=None,rgb_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005, training_syn=False):

    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv  = syn_data + '/test.csv'
    
    celeba_train_csv = None
    celeba_test_csv = None


    val_celeba_dl = None
    #tambahanku
    if celeba_data is not None:
        # celeba_train_csv = celeba_data + '/train.csv'
        # celeba_test_csv = celeba_data + '/test.csv'
        if training_syn:
            celeba_dt, _ = get_celeba_dataset(dir=celeba_data ,read_from_csv=None, read_first=batch_size, validation_split=99.9)
            val_celeba_dl = DataLoader(celeba_dt, batch_size=batch_size, shuffle=True)
    # Load Synthetic dataset
    train_dataset, val_dataset = get_sfsnet_dataset(read_from_csv=syn_train_csv, read_celeba_csv=celeba_train_csv ,read_first=read_first, validation_split=2, training_syn = training_syn)
    test_dataset, _ = get_sfsnet_dataset(read_from_csv=syn_test_csv, read_celeba_csv=celeba_test_csv,read_first=100, validation_split=0, training_syn = training_syn)

    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_val_dl    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Val data: ', len(syn_val_dl), ' Test data: ', len(syn_test_dl))
    
    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'val/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))
    if val_celeba_dl is not None:
        os.system('mkdir -p {}'.format(out_syn_images_dir + 'celeba_val/'))
        
    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=wt_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    normal_loss = nn.MSELoss()
    albedo_loss = nn.MSELoss()
    sh_loss     = nn.MSELoss()
    recon_loss  = nn.MSELoss() 

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()

    lamda_recon  = 1
    lamda_albedo = 1
    lamda_normal = 1
    lamda_sh     = 1

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()

    syn_train_len    = len(syn_train_dl)

    for epoch in range(1, num_epochs+1):
        tloss = 0 # Total loss
        nloss = 0 # Normal loss
        aloss = 0 # Albedo loss
        shloss = 0 # SH loss
        rloss = 0 # Reconstruction loss

        
        for bix, data in enumerate(syn_train_dl):
            albedo, normal, sh, face = data
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                sh     = sh.cuda()
                face   = face.cuda()
            # Apply Mask on input image
            # face = applyMask(face, mask)
            predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon = sfs_net_model(face)
            # Loss computation
            # Normal loss
            current_normal_loss = normal_loss(predicted_normal, normal)
            # Albedo loss
            current_albedo_loss = albedo_loss(predicted_albedo, albedo)
            # SH loss
            current_sh_loss     = sh_loss(predicted_sh, sh)

            current_recon_loss  = recon_loss(out_recon, face)

            total_loss = lamda_normal * current_normal_loss \
                           + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss #  + lamda_recon * current_recon_loss 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging for display and debugging purposes
            tloss += total_loss.item()
            nloss += current_normal_loss.item()
            aloss += current_albedo_loss.item()
            shloss += current_sh_loss.item()
            rloss += current_recon_loss.item()

        #scheduler.step(epoch)
        epoch_alert='Epoch: {} Learning rate: {} - Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(epoch , optimizer.param_groups[0]['lr'],\
                                tloss / syn_train_len, nloss / syn_train_len, aloss / syn_train_len, shloss / syn_train_len, rloss / syn_train_len)
        print(epoch_alert)
        with open(log_path+'details.txt', 'a') as f:
            f.write('\n'+epoch_alert)

        with torch.no_grad():
            log_prefix = 'Syn Data'
            if celeba_data is not None:
                log_prefix = 'Mix Data '

            # Model saving
            # torch.save({'epoch': epoch, 
            # 'model_state_dict': sfs_net_model.state_dict(), 
            # 'optimizer_state_dict': optimizer.state_dict()
            # }, model_checkpoint_dir + 'skipnet_model_'+str(epoch) +'.pkl')
            if epoch % 5 == 0:

                v_total, v_normal, v_albedo, v_sh, v_recon = predict_synthetic(sfs_net_model, syn_val_dl, train_epoch_num=epoch, use_cuda=use_cuda,
                                                                             out_folder=out_syn_images_dir+'/val/')
                val_alert='Val Synthetic set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(v_total,
                        v_normal, v_albedo, v_sh, v_recon)
                print(val_alert)
                #write to log
                with open(log_path+'details.txt', 'a') as f:
                    f.write('\n'+val_alert)

            if epoch % 10 == 0:
                t_total, t_normal, t_albedo, t_sh, t_recon = predict_synthetic(sfs_net_model, syn_test_dl, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                            out_folder=out_syn_images_dir + '/test/', suffix='Test')

                test_alert= 'Test-set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(t_total,
                                                                                                        t_normal, t_albedo, t_sh, t_recon)
                print(test_alert)

                #write to log
                with open(log_path+'details.txt', 'a') as f:
                    f.write('\n'+test_alert)

                if val_celeba_dl is not None:
                    lossku=predict_celeba(sfs_net_model, val_celeba_dl, train_epoch_num = epoch,
                            use_cuda = use_cuda, out_folder = out_syn_images_dir + 'celeba_val/',  dump_all_images = True)
                    real_test='Test-celeba set results: Total Loss:{}\n'.format(lossku)
                    print(real_test)
                     #write to log
                    with open(log_path+'details.txt', 'a') as f:
                        f.write('\n'+real_test)


def train(sfs_net_model, syn_data, celeba_data=None,read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005, training_syn=False, optim_state=None, last_epoch = None):
    
    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv  = syn_data + '/test.csv'
    
    celeba_train_csv = celeba_data + '/train.csv'
    celeba_test_csv = celeba_data + '/test.csv'

    # Load Synthetic dataset
    train_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data+'train/', read_from_csv=syn_train_csv, read_celeba_csv=celeba_train_csv, read_first=read_first, validation_split=0, training_syn = training_syn)
    test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data+'test/', read_from_csv=syn_test_csv, read_celeba_csv=celeba_test_csv,read_first=100, validation_split=0, training_syn = training_syn)

    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_test_dl),' Test data: ', len(syn_test_dl))
    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))
        

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=wt_decay)
    if (optim_state is not None ):
        optimizer.load_state_dict(optim_state)
        #optimizer.param_groups[0]['lr']=0.1
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.1, patience=2, verbose=True, threshold=0.1,threshold_mode='abs', min_lr= 0.0001)

    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss     = nn.MSELoss()
    recon_loss  = nn.L1Loss() 
    shading_loss = nn.L1Loss()

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss     = sh_loss.cuda()
        recon_loss  = recon_loss.cuda()
        shading_loss = shading_loss.cuda()

    lamda_recon  = 0.5
    lamda_albedo = 0.5
    lamda_normal = 0.5
    lamda_sh     = 0.1
    lamda_shading = 0.5


    syn_train_len    = len(syn_test_dl)
    start_epoch = 1

    if last_epoch is not None :
        start_epoch = last_epoch+1
        
    
    #with mlflow.start_run(experiment_id=0):
    for epoch in range(start_epoch, num_epochs+1):
        #sfs_net_model.train()
        tloss = 0 # Total loss
        nloss = 0 # Normal loss
        aloss = 0 # Albedo loss
        shloss = 0 # SH loss
        rloss = 0 # Reconstruction loss

        for bix, data in enumerate(syn_test_dl):
            albedo, normal, sh, face, faceHR = data
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                sh     = sh.cuda()
                face   = face.cuda()
                faceHR   = faceHR.cuda()
                shading = get_shading(normal, sh).cuda()
            else :
            	shading = get_shading(normal, sh)
            # Apply Mask on input image

            # a=face[0].view(3,128,128).permute(1,2,0)
            # a=a.detach().cpu().numpy()
            # #b=nn.functional.interpolate(faceHR, size=128)
            # b=normal[0].view(3,128,128).permute(1,2,0)
            # b=b.detach().cpu().numpy()
            # fig=plt.figure(figsize=(3, 3))
            # fig.add_subplot(131)
            # plt.imshow(a)
            # fig.add_subplot(132)
            # plt.imshow(b)     
            # plt.show()
            #Train Discriminator   
            predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon = sfs_net_model(faceHR)
            # Normal loss
            current_normal_loss = normal_loss(predicted_normal, normal)
            # Albedo loss
            current_albedo_loss = albedo_loss(predicted_albedo, albedo)
            # SH loss
            current_sh_loss     = sh_loss(predicted_sh, sh)
            # Reconstruction loss
            # Edge case: Shading generation requires denormalized normal and sh
            # Hence, denormalizing face here
            # Base
            current_recon_loss  = recon_loss(out_recon, face)

            total_loss = lamda_normal * current_normal_loss \
                           + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss + lamda_recon * current_recon_loss

            #base
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()   



            # Logging for display and debugging purposes
            tloss += total_loss.item()
            nloss += current_normal_loss.item()
            aloss += current_albedo_loss.item()
            shloss += current_sh_loss.item()
            rloss += current_recon_loss.item()


        train_alert ='Epoch: {} - Learning Rate: {}. Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(\
        epoch, optimizer.param_groups[0]['lr'] , tloss / syn_train_len, nloss / syn_train_len, aloss / syn_train_len, shloss / syn_train_len , rloss / syn_train_len)
        print(train_alert)
        with open(log_path+'train_details.txt', 'a') as f:
            f.write('\n'+str(epoch)+','+str(optimizer.param_groups[0]['lr'])+','+str(round(aloss / syn_train_len,4))+','+str(round(nloss / syn_train_len,4))+\
                ','+str(round(shloss / syn_train_len,4))+','+str(round(rloss / syn_train_len,4))+','+str(round(tloss / syn_train_len,4)))


        print("sampai test")
        with torch.no_grad():
            log_prefix = 'Syn Data'
            if celeba_data is not None:
                log_prefix = 'Mix Data '
            #Model saving
            # torch.save({'epoch': epoch,
            #     'model_state_dict': sfs_net_model.state_dict(), 
            #     'optimizer_state_dict': optimizer.state_dict()
            #     }, model_checkpoint_dir + 'sfs_net_model_'+str(epoch)+'.pkl')

            if epoch % 5 == 0:
                t_total, t_normal, t_albedo, t_sh, t_recon = predict_sfsnet(sfs_net_model, syn_test_dl, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                            out_folder=out_syn_images_dir + '/test/', wandb=wandb, suffix='Test')
                
                test_alert = 'Test-set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}\n'.format(t_total,
                                                                                                        t_normal, t_albedo, t_sh, t_recon)
                print(test_alert)
                with open(log_path+'test_details.txt', 'a') as f:
                    f.write('\n'+str(epoch)+','+str(optimizer.param_groups[0]['lr'])+','+str(round(t_albedo,4))+','+str(round(t_normal,4))+','+str(round(t_sh,4))+','+str(round(t_recon,4))+','+str(round(t_total,4)))
                print("selesai test")

def train_withPretrain(sfs_net_model, syn_data, celeba_data=None, rgb_syn_data=None, rgb_real_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005, training_syn=False, optim_state=None, last_epoch = None,  perceptual_loss =None):
    
    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv  = syn_data + '/test.csv'
    
    celeba_train_csv = celeba_data + '/train.csv'
    celeba_test_csv = celeba_data + '/test.csv'

    rgb_synt_train_csv = rgb_syn_data + '/rgb_train.csv'
    rgb_synt_test_csv = rgb_syn_data + '/rgb_test.csv'

    rgb_real_train_csv = rgb_real_data + '/rgb_real_train.csv'
    rgb_real_test_csv = rgb_real_data + '/rgb_real_test.csv'

    # Load Synthetic dataset
    train_dataset,_ = get_light_dataset(syn_dir=syn_data+'train/', read_from_csv=syn_train_csv, read_celeba_csv=celeba_train_csv,read_rgb_synt_csv=rgb_synt_train_csv, \
        read_rgb_real_csv=rgb_real_train_csv,read_first=read_first, validation_split=0, training_syn = training_syn)
    test_dataset, _ = get_light_dataset(syn_dir=syn_data+'test/', read_from_csv=syn_test_csv, read_celeba_csv=celeba_test_csv,read_rgb_synt_csv=rgb_synt_test_csv,\
        read_rgb_real_csv=rgb_real_test_csv,read_first=read_first, validation_split=0, training_syn = training_syn)

    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl),' Test data: ', len(syn_test_dl))
    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))
        

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=wt_decay)
    if (optim_state is not None ):
        optimizer.load_state_dict(optim_state)
        #optimizer.param_groups[0]['lr']=0.1
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.1, patience=2, verbose=True, threshold=0.1,threshold_mode='abs', min_lr= 0.0001)

    albedo_corrector_loss = nn.L1Loss()
    shading_corrector_loss = nn.MSELoss()
    recon_corrector_loss = nn.L1Loss()

    if use_cuda:
        shading_corrector_loss = shading_corrector_loss.cuda()
        albedo_corrector_loss  = albedo_corrector_loss.cuda()
        recon_corrector_loss = recon_corrector_loss.cuda()

    # lamda_recon  = 0.5
    # lamda_albedo = 0.5
    # lamda_normal = 0.5
    # lamda_sh     = 0.1
    # lamda_shading = 0.5


    syn_train_len    = len(syn_test_dl)
    start_epoch = 1

    if last_epoch is not None :
        start_epoch = last_epoch+1
        
    
    #with mlflow.start_run(experiment_id=0):
    for epoch in range(start_epoch, num_epochs+1):
        #sfs_net_model.train()
        alcorloss = 0 
        shcorloss = 0
        recorloss = 0

        tloss = 0

        for bix, data in enumerate(syn_test_dl):
            _, face, target = data
            if use_cuda:
                face = face.cuda()
                target = target.cuda()
            # Apply Mask on input image


            _, target_albedo, _, target_shading, target_recon, _, _,_ = sfs_net_model(target)         


            _, _, _, _, _, albedo_correct, shading_correct,recon_correct = sfs_net_model(face)

            # a=target_shading[0].view(3,256,256).permute(1,2,0)
            # a=a.detach().cpu().numpy()
            # b=target[0].view(3,256,256).permute(1,2,0)
            # b=b.detach().cpu().numpy()
            # fig=plt.figure(figsize=(3, 3))
            # fig.add_subplot(131)
            # plt.imshow(a)
            # fig.add_subplot(132)
            # plt.imshow(b)          
            # plt.show()

            #print(pytorch_ssim.ssim(shading_correct, target_shading))
            current_shading_cor_loss = shading_corrector_loss(shading_correct,target_shading)  +  0.2* perceptual_loss(shading_correct,target_shading) #pytorch_ssim.ssim(shading_correct, target_shading)
            current_albedo_cor_loss = albedo_corrector_loss(albedo_correct,target_albedo) +  0.8* perceptual_loss(albedo_correct,target_albedo)
            current_recon_cor_loss = recon_corrector_loss(recon_correct,target_recon) +  perceptual_loss(recon_correct,target_recon)

            total_loss = current_shading_cor_loss + current_albedo_cor_loss + current_recon_cor_loss

            #base
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()   

            # if bix==0:
            #     print(sfs_net_model.module.albedo_corrector_model.conv1[0].weight.grad[0])

            # Logging for display and debugging purposes
            tloss += total_loss.item()
            shcorloss += current_shading_cor_loss.item()
            alcorloss += current_albedo_cor_loss.item()
            recorloss +=current_recon_cor_loss.item()


        train_alert ='Epoch: {} - Learning Rate: {}. Total Loss: {}, Shading Corrector Loss: {}, Albedo Corrector Loss: {}, Recon Corrector Loss: {}'.format(\
        epoch, optimizer.param_groups[0]['lr'] , tloss / syn_train_len, shcorloss / syn_train_len, alcorloss / syn_train_len, recorloss/syn_train_len)
        print(train_alert)
        # with open(log_path+'train_details.txt', 'a') as f:
        #     f.write('\n'+str(epoch)+','+str(optimizer.param_groups[0]['lr'])+','+str(round(aloss / syn_train_len,4))+','+str(round(nloss / syn_train_len,4))+\
        #         ','+str(round(shloss / syn_train_len,4))+','+str(round(rloss / syn_train_len,4))+','+str(round(tloss / syn_train_len,4)))


        with torch.no_grad():
            log_prefix = 'Syn Data'
            if celeba_data is not None:
                log_prefix = 'Mix Data '
            #Model saving
            # torch.save({'epoch': epoch,
            #     'model_state_dict': sfs_net_model.state_dict(), 
            #     'optimizer_state_dict': optimizer.state_dict()
            #     }, model_checkpoint_dir + 'sfs_net_model_'+str(epoch)+'.pkl')

            if epoch % 5 == 0:
                t_albedo, t_shading,t_recon = predict_corrector(sfs_net_model,syn_test_dl, perceptual_loss, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                            out_folder=out_syn_images_dir + '/test/', wandb=wandb, suffix='Test')
                
                test_alert = 'Test-set results: Albedo Loss: {}, shading Loss: {} , Recon Loss: {}\n'.format(t_albedo, t_shading,t_recon)
                print(test_alert)
                # with open(log_path+'test_details.txt', 'a') as f:
                #     f.write('\n'+str(epoch)+','+str(optimizer.param_groups[0]['lr'])+','+str(round(t_albedo,4))+','+str(round(t_normal,4))+','+str(round(t_sh,4))+','+str(round(t_recon,4))+','+str(round(t_total,4)))
                # print("selesai test")

#Train for Original Skin Recovery Network
def train_new_style(generator,discriminator,rgb_syn_data=None, rgb_real_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005, training_syn=False, last_g_state=None, last_d_state=None, last_epoch = None, perceptual_loss=None):
    
    #Prepare to read from CSV
    rgb_synt_train_csv = rgb_syn_data + '/rgb_train.csv'
    rgb_synt_test_csv = rgb_syn_data + '/rgb_test.csv'

    rgb_real_train_csv = rgb_real_data + '/rgb_real_train.csv'
    rgb_real_test_csv = rgb_real_data + '/rgb_real_test.csv'


    val_celeba_dl = None

    # Load dataset
    train_dataset,_ = get_light_dataset(read_rgb_synt_csv=rgb_synt_train_csv,read_rgb_real_csv=rgb_real_train_csv,read_first=read_first, validation_split=0, training_syn = training_syn)
    test_dataset, _ = get_light_dataset(read_rgb_synt_csv=rgb_synt_test_csv,read_rgb_real_csv=rgb_real_test_csv,read_first=10, validation_split=0, training_syn = training_syn)
    # Create Dataloader
    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Test data: ', len(syn_test_dl))
    #Create Checkpoints & sample training result folder
    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))
    if val_celeba_dl is not None:
        os.system('mkdir -p {}'.format(out_syn_images_dir + 'celeba_val/'))
        

    # Collect model parameters
    generator_parameters = generator.parameters()
    discriminator_parameters = discriminator.parameters()
    G_optimizer = torch.optim.Adam(generator_parameters, lr=lr, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(discriminator_parameters,lr=lr, betas=(0.5, 0.999))
    # Load last checkpoint if want to continue
    if (last_g_state is not None ):
        G_optimizer.load_state_dict(last_g_state)
        D_optimizer.load_state_dict(last_d_state)

    #Create Loss
    perceptual_loss =perceptual_loss
    image_loss = nn.L1Loss()

    #assign Loss to CUDA
    if use_cuda:
        image_loss = image_loss.cuda()

    #Hyperparameter tuning
    lambda_perceptual = 1.0
    lambda_gp = 10.0
    lambda_recon = 10.0
    lambda_adv = 0.1


    syn_train_len    = len(syn_train_dl)
    start_epoch = 1
    if last_epoch is not None :
        start_epoch = last_epoch+1

    #i = Iteration
    i = 0
    best_record = 0
    for epoch in range(start_epoch, num_epochs+1):
        t_d_loss_fake = 0
        t_d_loss_real = 0
        t_d_loss_gp = 0
        t_dloss = 0
        t_g_loss_fake = 0
        t_g_loss_real = 0
        t_g_loss_recon = 0
        t_gloss = 0
        t_g_loss_perceptual=0

        for bix, data in enumerate(syn_train_dl):
            face, target = data
            if use_cuda:
                face   = face.cuda()
                target   = target.cuda()

            #Train Discriminator  
            out_target = discriminator(target)
            d_loss_real = - torch.mean(out_target)

            x_fake = generator(face)
            out_src = discriminator(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(target.size(0), 1, 1, 1).cuda()
            x_hat = (alpha * target.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src = discriminator(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat)


            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + lamda_gp*d_loss_gp
            D_optimizer.zero_grad()
            d_loss.backward()
            D_optimizer.step()   

            t_d_loss_fake += d_loss_fake.item()
            t_d_loss_real += d_loss_real.item()
            t_dloss +=d_loss.item()

            
            #Train Generator
            fake_image = generator(face)
            out_src = discriminator(fake_image)

            #Compute Loss
            g_loss_fake = - torch.mean(out_src)
            g_loss_perceptual = perceptual_loss(fake_image,target)
            g_loss_recon = image_loss(fake_image,target)


            # Backward and optimize.
            g_loss =  lambda_perceptual*g_loss_perceptual +  lambda_recon*g_loss_recon + lambda_adv*g_loss_fake
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()


            t_g_loss_fake += g_loss_fake.item()
            t_g_loss_perceptual += g_loss_perceptual.item()
            t_g_loss_recon += g_loss_recon.item()
            t_gloss += g_loss.item()

            # Logging for display and debugging purposes
            i=i+1
            # Print result every 100 iteration
            if i % 100 == 0:

                train_alert ='Iteration: {} Generator Loss: {}, Fake Generator Loss :{}, perceptual Loss : {} , Recon Loss : {}, Discriminator Loss : {}, real-discriminator : {}, fake-discriminator : {}'.format(\
                i, g_loss_fake.item()+g_loss_perceptual.item()+ g_loss_recon.item(), g_loss_fake.item() , g_loss_perceptual.item(), g_loss_recon.item(),  \
                d_loss_fake.item()+d_loss_real.item(),d_loss_real.item(), d_loss_fake.item() )
                print(train_alert)

                with open(log_path+'val_details.txt', 'a') as f:
                    f.write('\n'+str(i)+','+str(round(g_loss_fake.item()+g_loss_perceptual.item()+ g_loss_recon.item(),4))+','+str(round(g_loss_fake.item(),4))
                        +','+str(round(g_loss_perceptual.item(),4)) +','+str(round(g_loss_recon.item(),4))\
                         +','+str(round(g_loss_fake.item()+d_loss_real.item(),4)) +','+str(round(d_loss_real.item(),4)) +','+str(round(d_loss_fake.item(),4)))

                with torch.no_grad():
                    t_cor = predict_light(generator,syn_test_dl,perceptual_loss, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                                out_folder=out_syn_images_dir + '/test/', iteration = i, suffix='Test')
                    
                    test_alert = 'Iteration {} ---- Test-set results:  Cor Loss {}'.format(i,t_cor)
                    print(test_alert)

        # Print result every epoch
        train_alert ='Epoch: {} Generator Loss: {}, Fake Generator Loss :{}, perceptual Loss : {} , Recon Loss : {}, Discriminator Loss : {}, real-discriminator : {}, fake-discriminator : {}'.format(\
        epoch, t_gloss / syn_train_len, t_g_loss_fake/syn_train_len , t_g_loss_perceptual/syn_train_len, t_g_loss_recon/syn_train_len,  t_dloss/syn_train_len,t_d_loss_real/syn_train_len, t_d_loss_fake/syn_train_len )
        print(train_alert)
        with open(log_path+'details.txt', 'a') as f:
            f.write('\n'+str(epoch)+','+str(round(t_gloss / syn_train_len,4))+','+str(round(t_g_loss_fake/syn_train_len,4))
                +','+str(round(t_g_loss_perceptual/syn_train_len,4)) +','+str(round(t_g_loss_recon/syn_train_len,4))\
                 +','+str(round(t_dloss/syn_train_len,4)) +','+str(round(t_d_loss_real/syn_train_len,4)) +','+str(round( t_d_loss_fake/syn_train_len,4)))

        #save the network
        with torch.no_grad():
            log_prefix = 'Syn Data'
            if celeba_data is not None:
                log_prefix = 'Mix Data '
            #Model saving
            torch.save({'epoch': epoch,
                'generator_state_dict': generator.state_dict(), 
                'discriminator_state_dict': discriminator.state_dict(), 
                'g_optimizer_dict': G_optimizer.state_dict(),
                'd_optimizer_dict': D_optimizer.state_dict(),
                }, model_checkpoint_dir + 'light_gan'+str(epoch)+'.pkl')

