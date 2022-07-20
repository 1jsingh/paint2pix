import numpy as np
from collections import deque
import cv2
import pandas as pd
import os,sys
import glob

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from PIL import Image

from utils import id_utils

from utils.common import tensor2im
from models.psp import pSp
from models.e4e import e4e
from utils.inference_utils import run_on_batch
from criteria import id_loss, moco_loss
import streamlit as st

from argparse import Namespace
import torch
import clip
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_model(experiment_type='ffhq',id_constrain=False,stylized_output=False):
    with torch.no_grad():
        if experiment_type == 'ffhq':
            if id_constrain:
                model_path = 'pretrained_models/id-encoder.pt'
                is_canvas_encoder = False
                input_nc = 6
            else:
                model_path = 'pretrained_models/canvas_encoder.pt'
                is_canvas_encoder = True
                input_nc = 12

            resize_dims = (256,256)
        
        elif experiment_type == 'cars_encode':
            model_path = 'experiments/cars196/intelli-paint/paint_512_v1/checkpoints/best_model.pt'
            resize_dims = (192,256)
        
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']

        # update the training options
        opts['checkpoint_path'] = model_path
        opts['input_nc'] = input_nc
        opts['is_canvas_encoder'] = is_canvas_encoder

        opts = Namespace(**opts)
        net = e4e(opts)
        
        if stylized_output:
            decoder_path = 'pretrained_models/stylegan2-watercolor.pt'
            decoder_ckpt = torch.load(decoder_path)
            net.decoder.load_state_dict(decoder_ckpt, strict=True)
        
        net.eval()
        net = net.to(device)
        print('Model successfully loaded!')
        
        transform = transforms.Compose([
				transforms.Resize(resize_dims),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    return net, transform, opts


def display_alongside_source_image(result_image, source_image):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)

def display_alongside_batch(img_list, resize_dims):
    res = np.concatenate([np.array(img.resize(resize_dims)) for img in img_list], axis=1)
    return Image.fromarray(res)

def get_avg_image(net, experiment_type='ffhq'):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    if experiment_type == "cars_encode":
        avg_image = avg_image[:, 32:224, :]
    return avg_image

def get_multi_modal_outputs(x, net, vectors_to_inject, latent_mask=[0,1], mix_alpha=None, input_code=False):
    results = []
    with torch.no_grad():
        for vec_to_inject in vectors_to_inject:
            cur_vec = torch.from_numpy(vec_to_inject).unsqueeze(0).to("cuda")
            # get latent vector to inject into our input image
            _, latent_to_inject = net(cur_vec,
                                    input_code=True,
                                    return_latents=True)
            
            if input_code:
                inject_latent = latent_to_inject
                alpha = mix_alpha
                codes = x
                # get latents
                # print (inject_latent.shape,codes.shape)
                if latent_mask is not None:
                    for i in latent_mask:
                        if inject_latent is not None:
                            if alpha is not None:
                                codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                            else:
                                codes[:, i] = inject_latent[:, i]
                        else:
                            codes[:, i] = 0

                input_is_latent = input_code
                # st.text(codes.shape)
                images, result_latent = net.decoder([codes],
                                                    input_is_latent=input_is_latent,
                                                     randomize_noise=False,
                                                     return_latents=False)
                res = net.face_pool(images)
                # st.text(res.shape)
            else:
                # get output image with injected style vector
                res = net(x.unsqueeze(0).to("cuda").float(),
                        latent_mask=latent_mask,
                        inject_latent=latent_to_inject,
                        alpha=mix_alpha)
            results.append(res[0])
    return results

def predict_image_completion(image, net, transform, opts, experiment_type='ffhq', resize_dims=(256,256), multi_modal=False, num_multi_output=5, n_iters=5, latent_mask=None ,mix_alpha=None, id_constrain=False, target_id_feat=None):
    opts.n_iters_per_batch = n_iters
    opts.resize_outputs = False  # generate outputs at full resolution
    transformed_image = transform(image).to(device)
    with torch.no_grad():
        avg_image = get_avg_image(net,experiment_type)
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net, opts, avg_image)
        #run_on_batch(transformed_image.unsqueeze(0), net, experiment_type=experiment_type)
    result_images, latent = images[0], latents[0]
    result_images = [tensor2im(result_images[iter_idx]).resize(resize_dims[::-1]) for iter_idx in range(opts.n_iters_per_batch)]
    #result_image = tensor2im(result_image)
    
    if multi_modal:
        # randomly draw the latents to use for style mixing
        vectors_to_inject = np.random.randn(num_multi_output, 512).astype('float32')
        
        with torch.no_grad():
            latent = torch.tensor(latent[-1]).to("cuda").float().unsqueeze(0)
            multi_results = get_multi_modal_outputs(latent, net, vectors_to_inject, latent_mask, mix_alpha, input_code=True)
            # out = net.decoder(latents,input_is_latent=True)[0].detach().cpu()
        img_list = [result_images[-1]] + [tensor2im(x).resize(resize_dims[::-1]) for x in multi_results]
        res = display_alongside_batch(img_list[0:],resize_dims)
        return result_images, res, latents
    else:
        return result_images, None, latents
    

def decode_latent(x, net, opts, experiment_type='ffhq', resize_dims=(256,256),convert2im=True, preprocess=True, truncation=1, truncation_latent=None):
    with torch.no_grad():
        if preprocess:
            latent = torch.tensor(x).to("cuda").float().unsqueeze(0)
        else:
            latent = x
        
        if truncation_latent is None and truncation < 1:
            truncation_latent = net.decoder.mean_latent(4096)
        
        images, result_latent = net.decoder([latent],
                                            input_is_latent=True,
                                             randomize_noise=False,
                                             return_latents=False,
                                            truncation=truncation,
                                            truncation_latent=truncation_latent)
        res = net.face_pool(images)
        if convert2im:
            res = tensor2im(res[0]).resize(resize_dims[::-1])
    return res


def decode_latent2(x, net, opts, experiment_type='ffhq', resize_dims=(256,256),convert2im=True, preprocess=True):
    latent= x
    images, result_latent = net.decoder([latent],
                                        input_is_latent=True,
                                         randomize_noise=False,
                                         return_latents=False)
    res = net.face_pool(images)
    return res



@st.cache
def load_faceid_model():
    id_loss_func = id_loss.IDLoss().to(device).eval()
    return id_loss_func



def encoder_based_id_edit(original_img, initial_latent, target_img, net, transform, opts, latent_mask=None, num_id_iter=5):
    initial_latent = torch.tensor(initial_latent).to("cuda").float().unsqueeze(0)
    
    if latent_mask is None:
        latent_mask = np.ones(18)
    mask = torch.tensor(latent_mask).float().repeat((512,1)).transpose(1,0).unsqueeze(0).to(device)
    mask.requires_grad = False
    
    with torch.no_grad():
        #avg_image_for_batch, initial_latent = id_utils.predict_image_completion(y, id_net, transform, opts)
        #y = avg_image_for_batch
        avg_image_for_batch = transform(original_img).to(device).unsqueeze(0)
        x = transform(target_img).to(device).unsqueeze(0)
        
        latent1 = initial_latent
        for iter in range(num_id_iter):
            target_id_feat = None#self.id_loss.extract_feats(id_x)
            if iter == 0:
                x_input = torch.cat([x, avg_image_for_batch], dim=1)
            else:
                x_input = torch.cat([x, y_hat], dim=1)
                
            y_hat, latent2 = net.forward(x_input, target_id_feat=target_id_feat, latent=latent1, return_latents=True)
            latent2 = latent1 + (latent2 - latent1)*mask
            latent1 = latent2
            
            if opts.dataset_type == "cars_encode":
                y_hat = y_hat[:, :, 32:224, :]
                
    out_img = decode_latent(latent2,net, opts, preprocess=False) #tensor2im(y_hat[0])
    
    return out_img, (latent2-initial_latent).detach().cpu().numpy()

    
def identity_constrained_latent_pred(x, target_img, net, transform, opts, id_loss_func, input_code=True, n_iter=20, lr=1e-3, lambda_reg=0.5,lambda_id=1.0,lambda_l2=1e-1,latent_mask=None):
    if input_code:
        w0 = torch.tensor(x).to("cuda").float().unsqueeze(0)
    
    delta_w = torch.zeros_like(w0,requires_grad=True).to(device).float()
    opt = optim.Adam([delta_w],lr)
    
    if latent_mask is None:
        latent_mask = np.ones(18)
    mask = torch.tensor(latent_mask).float().repeat((512,1)).transpose(1,0).unsqueeze(0).to(device)
    mask.requires_grad = False
    
    test = False
    if test:
        w_ = torch.tensor(target_img).to("cuda").float().unsqueeze(0)
        w1 = mask * w_ + (1-mask) * w0
        n_iter = 0
    
    
    img0 = decode_latent(w0, net, opts, convert2im=False,preprocess=False)
    img0.requires_grad = False
    
    target_img = transform(target_img).to(device).unsqueeze(0)
    target_img.requires_grad = False
    
    loss_dict = {'id_loss':[],'reg_loss':[],'l2_loss':[], 'total_loss':[]}
    
    my_bar = st.progress(0)
    for i in range(n_iter):
        my_bar.progress(int(100*(i+1)/n_iter))
        w1 = w0 + delta_w * mask
        img1 = decode_latent2(w1, net, opts, convert2im=False,preprocess=False)
        
        loss_id, _, _ = id_loss_func(img1,target_img,target_img)
        loss = lambda_id * loss_id
        loss_dict['id_loss'] += [loss_id.item()]
        
        l2_loss = F.mse_loss(img1, img0)
        loss += lambda_l2 * l2_loss
        loss_dict['l2_loss'] += [lambda_l2 * l2_loss.item()]
        
        reg_loss = torch.norm(delta_w)  #F.mse_loss(delta_w, 0)
        loss += lambda_reg * reg_loss
        loss_dict['reg_loss'] += [reg_loss.item()]
        
        # compute total loss
        loss_dict['total_loss'] += [loss.item()]
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # plot loss
    idx_list = ['id_loss', 'reg_loss', 'l2_loss', 'total_loss']
    chart_data = pd.DataFrame( np.stack([loss_dict[x] for x in idx_list],axis=-1) , columns=idx_list)
    
    out_img = decode_latent(w1, net, opts, convert2im=True,preprocess=False)
    return out_img, chart_data, delta_w.detach().cpu().numpy()