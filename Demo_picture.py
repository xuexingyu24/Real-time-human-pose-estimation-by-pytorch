#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:41:57 2019
This is an easy version to speed up the inference 
* perform grouping on input image to net, while not original image 

@author: AIRocker
"""
import cv2
from model import bodypose_model, PoseEstimationWithMobileNet
from utils.util import*
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import argparse

def Net_Prediction(model, image, device, backbone = 'Mobilenet'):
    scale_search = [1]
    stride = 8
    padValue = 128
    heatmap_avg = np.zeros((image.shape[0], image.shape[1], 19))
    paf_avg = np.zeros((image.shape[0], image.shape[1], 38))
    
    for m in range(len(scale_search)):
        scale = scale_search[m]
        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
        # pad right and down corner to make sure image size is divisible by 8
        im = np.transpose(np.float32(imageToTest_padded), (2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)
   
        with torch.no_grad():
            if backbone == 'CMU':
                Mconv7_stage6_L1, Mconv7_stage6_L2 = model(data)
                _paf = Mconv7_stage6_L1.cpu().numpy()
                _heatmap = Mconv7_stage6_L2.cpu().numpy()
            elif backbone == 'Mobilenet':
                stages_output = model(data)
                _paf = stages_output[-1].cpu().numpy()
                _heatmap = stages_output[-2].cpu().numpy()  
            
        # extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(_heatmap), (1, 2, 0))  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        #print(heatmap.shape)
        
        paf = np.transpose(np.squeeze(_paf), (1, 2, 0))  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        #print(paf.shape)
        heatmap_avg += heatmap / len(scale_search)
        paf_avg += paf / len(scale_search)
        
    return heatmap_avg, paf_avg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Open Pose Demo')
    parser.add_argument("-backbone", help='CMU, Mobilenet', default='Mobilenet', type=str)
    parser.add_argument("-image", help='image path', default='images/ski.jpg', type=str)
    parser.add_argument("-scale", help='scale to image',default=0.3, type=float)
    parser.add_argument("-show", nargs='+', help="types to show: -1 shows the skeletons, or idx for specific part",default = (-1, 0), type=int)   
    parser.add_argument("-thre", help="threshold for heatmap part",default=0.1, type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
#    device = torch.device("cpu")
    if args.backbone == 'CMU':
        model = bodypose_model().to(device)
        model.load_state_dict(torch.load('weights/bodypose_model', map_location=lambda storage, loc: storage))
    elif args.backbone == 'Mobilenet':
        model = PoseEstimationWithMobileNet().to(device)
        model.load_state_dict(torch.load('weights/MobileNet_bodypose_model', map_location=lambda storage, loc: storage))
    
    model.eval()
    print('openpose {} model is successfully loaded...'.format(args.backbone))
    
    test_image = args.image
    image = cv2.imread(test_image)
    imageToTest = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    since = time.time()
    
    heatmap, paf = Net_Prediction(model, imageToTest, device, backbone = args.backbone)
    
    t1 = time.time()
    print("model inference in {:2.3f} seconds".format(t1 - since))
    
    all_peaks = peaks(heatmap, args.thre)
    
    t2 = time.time()
    print("find peaks in {:2.3f} seconds".format(t2 - t1))
    
    if args.show[0] == -1:
        connection_all, special_k = connection(all_peaks, paf, imageToTest)
        
        t2 = time.time()
        print("find connections in {:2.3f} seconds".format(t2 - t1))
        
        candidate, subset = merge(all_peaks, connection_all, special_k)
            
        t3 = time.time()
        print("merge in {:2.3f} seconds".format(t3 - t2))
        
        canvas = draw_bodypose(image, candidate, subset, args.scale)
            
    else:
        canvas = draw_part(image, all_peaks, args.show, args.scale)
    
    print("total inference in {:2.3f} seconds".format(time.time() - since))
    
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()

