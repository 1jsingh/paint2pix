from utils.common import tensor2im
from PIL import ImageColor
import torch
import cv2

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

from utils.common import tensor2im
from models.psp import pSp
from models.e4e import e4e
# from utils.inference_utils import run_on_batch

from criteria import id_loss, moco_loss

# from utils.common import tensor2im
# from options.train_options import TrainOptions
# from models.psp import pSp

import streamlit as st


from argparse import Namespace

import torch
import clip
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(experiment_type='ffhq',use_baseline=False,id_constrain=False):
    with torch.no_grad():
        if experiment_type == 'ffhq':
            if use_baseline:
                model_path = 'pretrained_models/restyle_e4e_ffhq_encode.pt'
            else:
                #model_path = 'experiment_paint_v2/checkpoints/best_model.pt'
                #model_path = 'experiment_paint_v4/checkpoints/best_model.pt'
                #model_path = 'experiment_1024_v4/checkpoints/best_model.pt'
                if id_constrain:
                    model_path = 'experiments/celeba/intelli-paint/paint_1024_id-constrain_v2/checkpoints/best_model.pt'
                else:
                    model_path = 'experiments/celeba/intelli-paint/paint_1024_v1/checkpoints/best_model.pt'

            resize_dims = (256,256)
        
        elif experiment_type == 'cars_encode':
            model_path = 'pretrained_models/restyle_e4e_cars_encode.pt'
            model_path = 'experiments/cars196/intelli-paint/paint_512_v1/checkpoints/best_model.pt'
            resize_dims = (192,256)
        
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        # pprint.pprint(opts)  # Display full options used
        # update the training options
        opts['checkpoint_path'] = model_path
        opts['device'] = device
        
        opts = Namespace(**opts)
        net = e4e(opts)
        # if experiment_type == 'horse_encode' or experiment_type == 'ffhq_encode': 
        #     net = e4e(opts)
        # else:
        #     net = pSp(opts)
        
        net.eval()
        net = net.to(device)
        print('Model successfully loaded!')
        
        transform = transforms.Compose([
				transforms.Resize(resize_dims),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    return net, transform, opts


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

def run_on_batch(inputs, net, opts, avg_image, target_id_feat=None):
    y_hat, latent = None, None
    #results_batch = {idx: [] for idx in range(inputs.shape[0])}
    #results_latent = {idx: [] for idx in range(inputs.shape[0])}
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        y_hat, latent = net.forward(x_input,
                                    target_id_feat=target_id_feat,
                                    latent=latent,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=opts.resize_outputs)

        if opts.dataset_type == "cars_encode":
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        # # store intermediate outputs
        # for idx in range(inputs.shape[0]):
        #     results_batch[idx].append(y_hat[idx])
        #     results_latent[idx].append(latent[idx].cpu().numpy())

        # resize input to 256 before feeding into next iteration
        if opts.dataset_type == "cars_encode":
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)

    return y_hat, latent #results_batch, results_latent

def predict_image_completion(image, net, transform, opts, preprocess=False, experiment_type='ffhq', resize_dims=(256,256), multi_modal=False, num_multi_output=5, n_iters=5, latent_mask=None ,mix_alpha=None, id_constrain=False, target_id_feat=None):
    opts.n_iters_per_batch = n_iters
    opts.resize_outputs = False  # generate outputs at full resolution
    
    if preprocess:
        image = transform(image).to(device).unsqueeze(0)
    
    with torch.no_grad():
        avg_image = get_avg_image(net,experiment_type)
        images, latents = run_on_batch(image, net, opts, avg_image)
        #run_on_batch(transformed_image.unsqueeze(0), net, experiment_type=experiment_type)
    #result_images, latent = images[0], latents[0]
    
    if preprocess:
        result_image = tensor2im(result_images[-1]).resize(resize_dims[::-1])
    return images, latents