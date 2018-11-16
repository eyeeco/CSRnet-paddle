# -*- coding: UTF-8 -*-
import random
import os
from PIL import Image,ImageFilter,ImageDraw,ImageStat
import numpy as np
import h5py
import cv2
import argparse
import json
import time

def data_reader(json_path):
    def reader():
        with open(json_path, 'r') as outfile:        
            train_list = json.load(outfile)
            for item in train_list:
                img_path = item
                gt_path = img_path.replace('.png','.h5').replace('.jpg','.h5').replace('train','truth')
                img = Image.open(img_path).convert('RGB')
                gt_file = h5py.File(gt_path)
                target = np.asarray(gt_file['density'])
                if False:
                    crop_size = (img.size[0]/2,img.size[1]/2)
                    if random.randint(0,9)<= -1:
                        dx = int(random.randint(0,1)*img.size[0]*1./2)
                        dy = int(random.randint(0,1)*img.size[1]*1./2)
                    else:
                        dx = int(random.random()*img.size[0]*1./2)
                        dy = int(random.random()*img.size[1]*1./2)
                    img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
                    target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
                    if random.random()>0.8:
                        target = np.fliplr(target)
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                target = cv2.resize(target,(68,120),interpolation = cv2.INTER_CUBIC)*64
                img = np.asarray(img)
                img = cv2.resize(img,(540,960),interpolation = cv2.INTER_CUBIC)
                img = img/255.0
                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                for i in range(3):
                    img[:,:,i]=(img[:,:,i]-mean[i])/std[i]
#                 print("img:\n",img,"target:\n",target)
                img=img.reshape((1,-1))
                target=target.reshape((1,-1))
                yield img,target
    return reader
def infer_data_reader(json_path):
    def reader():
        with open(json_path, 'r') as outfile:        
            train_list = json.load(outfile)
            for item in train_list:
                img_path = item
                img = Image.open(img_path).convert('RGB')
                img = np.asarray(img)
                img = cv2.resize(img,(540,960),interpolation = cv2.INTER_CUBIC)
                img = img/255.0
                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                for i in range(3):
                    img[:,:,i]=(img[:,:,i]-mean[i])/std[i]
                img=img.reshape((1,-1))
                yield img
    return reader

def nomal(x):
    return (x-450) * 2 