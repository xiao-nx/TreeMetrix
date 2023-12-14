# -*- coding: UTF-8 -*- 
import os
import cv2
import math
import numpy as np
import tensorflow as tf

import imtrans

_NUM_DEG = 8
_NUM_MLS_WARP = 1
_NUM_PERPECTIVE_WARP = 1

_WIDTH = 1024
_HEIGHT = 1024

def decode_label(label):
  decode_label = np.zeros((label.shape[0], label.shape[1]), dtype=label.dtype)

  blue_idx = np.where(
    np.logical_and(label[:, :, 0] > 245, label[:, :, 1] < 10, label[:, :, 2] < 10))
  red_idx = np.where(
    np.logical_and(label[:, :, 0] < 10, label[:, :, 1] < 10, label[:, :, 2] > 245))
  decode_label[blue_idx] = 1
  decode_label[red_idx] = 2
  return decode_label


#保存图像
def saveDstImage(image,label,images_output,labels_output,count):

  #ROI
  image_ROI,label_ROI = imtrans.get_image_ROI(image, label)
  #resize
  image_resized = cv2.resize(image_ROI, (_WIDTH, _HEIGHT), interpolation=cv2.INTER_NEAREST)
  label_resized = cv2.resize(label_ROI, (_WIDTH, _HEIGHT), interpolation=cv2.INTER_NEAREST)
  #label
  label_decode = decode_label(label_resized)

  cv2.imwrite(images_output + 'image_{}.jpg'.format(count), image_resized)
  cv2.imwrite(labels_output + 'label_{}.jpg'.format(count), label_resized)

  return


def create_augmented_images(data_path_dict):
    
    image_dir = data_path_dict["image_input"]
    label_dir = data_path_dict["label_input"]
    images_output = data_path_dict["image_output"]
    labels_output = data_path_dict["label_output"]

    if not os.path.exists(images_output):
        os.makedirs(images_output)
    if not os.path.exists(labels_output):
        os.makedirs(labels_output)
    
    count = 0

    images_list = os.listdir(image_dir)
    
    for idx,fname in enumerate(images_list):
        image_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, fname).replace('image','label')

        print(image_path)
        image_bgr = cv2.imread(image_path)
        image_label = cv2.imread(label_path)
        image_no_white_boder,label_no_white_boder = imtrans.remove_white_boder(image_bgr,image_label)

        saveDstImage(image_no_white_boder, label_no_white_boder,images_output,labels_output,count)

        degree = math.floor(360 /_NUM_DEG)
        count += 1
        for i in range(_NUM_DEG):
            #rotate
            rotated_image, rotated_label = imtrans.get_rotated_image(image_no_white_boder,label_no_white_boder, (i+1)* degree)
            saveDstImage(rotated_image, rotated_label,images_output,labels_output,count)           
            count += 1
              
            for j in range(_NUM_PERPECTIVE_WARP):
              #perspective
              perspective_image, perspective_label = imtrans.get_perspective_image(rotated_image, rotated_label)
              saveDstImage(perspective_image, perspective_label,images_output,labels_output,count)
              count += 1
              
              for k in range(_NUM_MLS_WARP):
                  #mlswarp
                  mlswarp_image, mlswarp_label = imtrans.get_mlswarp_image(perspective_image, perspective_label)
                  saveDstImage(mlswarp_image, mlswarp_label,images_output,labels_output,count)
                  count += 1
                  print(count)

    print('total train sample count={}'.format(count))
    return
 


if __name__ == "__main__":
  data_path_dict = {
    "image_input" : "../data_raw/images/validation/",
    "label_input" : "../data_raw/annotations/validation/",
    "image_output" : "../data_augmented/images/validation/",
    "label_output" : "../data_augmented/annotations/validation/",
    }
  create_augmented_images(data_path_dict)


