# -*- coding: UTF-8 -*- 
import math
import random
import copy
import cv2
import numpy as np

import mlswarp



_Debug = False
_HEIGHT = 1024
_WIDTH = 1024

def remove_white_boder(image,label):

    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    idx = np.where(image_gray != 255)
    x_idx_list = idx[0]
    y_idx_list = idx[1]
    x1 = min(x_idx_list) + 10
    x2 = max(x_idx_list) - 10
    y1 = min(y_idx_list) + 10
    y2 = max(y_idx_list) - 10
    image_no_white_boder = image[x1:x2,y1:y2,:]
    label_no_white_boder = label[x1:x2,y1:y2,:]
    
    return image_no_white_boder,label_no_white_boder


def get_mlswarp_image(image, label, x_num_grid=4, y_num_grid=4):
    if (image.dtype != np.uint8):
        image = np.array(image, dtype=np.uint8)
    
    warp_image, warp_label = mlswarp.mlswarp(image, label, x_num_grid, y_num_grid)
    if _Debug:
        cv2.namedWindow('warp_image',0)
        cv2.imshow('warp_image',warp_image)
        cv2.waitKey()
    return warp_image, warp_label

def get_perspective_image(image,label):
    height = image.shape[0]
    width = image.shape[1]
    
    ratio_x = int(0.05*width)
    ratio_y = int(0.05*height)
    x1 = random.randint(0,ratio_x)
    x2 = random.randint(0,ratio_x)
    x3 = random.randint(0,ratio_x)
    x4 = random.randint(0,ratio_x)
    y1 = random.randint(0,ratio_y)
    y2 = random.randint(0,ratio_y)
    y3 = random.randint(0,ratio_y)
    y4 = random.randint(0,ratio_y)
    
    src_points = np.float32([[x1,y1],[x2,height-y2],[width-x3,height-y3],[width-1-x4,y4]])
    
    dst_points = np.float32([[0,0],[0,height-1],[width-1,height-1],[width-1,0]])
    perspective_matrix = cv2.getPerspectiveTransform(src_points,dst_points)

    image_perspective = cv2.warpPerspective(image, perspective_matrix, (width,height))    
    label_perspective = cv2.warpPerspective(label, perspective_matrix, (width,height))
     
    return image_perspective,label_perspective

def get_rotated_image(image, label, angle, center=None, scale=1.0):
    if angle % 90 == 0:
      k = angle / 90
      image_rotated = np.rot90(image, k)
      label_rotated = np.rot90(label, k)
      return image_rotated, label_rotated
    
    height = image.shape[0]
    width = image.shape[1]

    if center is None:
        center = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    #compute the new bounding dimensions of the image
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    #adjust the rotation matrix to take into account translation
    M[0,2] += (new_width /2 ) - (width / 2)
    M[1,2] += (new_height / 2) - (height / 2)
    
    
    image_rotated = cv2.warpAffine(image, M, (new_width, new_height),borderValue=(20,20,20))
    label_rotated = cv2.warpAffine(label, M, (new_width, new_height),borderValue=(20,20,20))

    return image_rotated, label_rotated


#获取图像中的年轮区域
def get_image_ROI(image,label):

    image_blue = label[:,:,0]
    image_green = label[:,:,1]
    image_red = label[:,:,2] #BGR
    idx = np.where(np.logical_and(image_red >= 250,image_green<5,image_blue<5))
    x_idx_list = idx[1]
    y_idx_list = idx[0]
    x_min = min(x_idx_list)
    x_max = max(x_idx_list)
    y_min = min(y_idx_list)
    y_max = max(y_idx_list)

    image_ROI = image[y_min:y_max, x_min:x_max, :]
    label_ROI = label[y_min:y_max, x_min:x_max, :]

    if _Debug:
        cv2.namedWindow('image_ROI',0)
        cv2.namedWindow('label_ROI',0)
        cv2.imshow('image_ROI',image_ROI)
        cv2.imshow('label_ROI',label_ROI)
        cv2.waitKey()
    
    return image_ROI,label_ROI

#HSV色彩变换
def randomHueSaturationValue(image,label, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
    hue_shift = np.uint8(hue_shift)
    h += hue_shift
    sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
    s = cv2.add(s, sat_shift)
    val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
    v = cv2.add(v, val_shift)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        

    return image,label
#高斯噪声
def addGaussianNoise(image,label,percetage=0.3):
    G_noise_img = copy.deepcopy(image)
    height = image.shape[0]
    width = image.shape[1]
    G_NoiseNum = int(percetage*height*width)
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,height)
        temp_y = np.random.randint(0,width)
        G_noise_img[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]

    return G_noise_img,label

#椒盐噪声
def addSaltAndPepperNoise(image,label,percetage=0.3):
    SP_noise_img = copy.deepcopy(image)
    height = image.shape[0]
    width = image.shape[1]
    SP_NoiseNum = int(percetage*height*width)
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0,height - 1)
        randG = np.random.randint(0,width - 1)
        randB = np.random.randint(0,3)

        if np.random.randint(0,1)==0:
            SP_noise_img[randR,randG,randB] = 0
        else:
            SP_noise_img[randR,randG,randB] = 255

    return SP_noise_img,label

if __name__ == '__main__':

    image = cv2.imread('image001.jpg')
    label = cv2.imread('label001.jpg')
    print(image.dtype)

    image1 = remove_white_boder(image)
    label1 = remove_white_boder(label)
    print(image1.dtype)
    cv2.imwrite('images.jpg',image)

    
    image_ROI,label_ROI = get_image_ROI(image1,label1)
    cv2.imwrite('image_ROI.jpg', image_ROI)
    print(image_ROI.dtype)

    img_pers,label_pers = get_perspective_image(image_ROI,label_ROI)
    cv2.imwrite('img_pers.jpg', img_pers)
    
    image_mls,label_mls = get_mlswarp_image( img_pers,label_pers)   
    cv2.imwrite('image_mls.jpg', image_mls)
    
