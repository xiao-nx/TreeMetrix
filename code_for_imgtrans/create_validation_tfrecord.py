#coding:utf-8

# 打包数据，不作预处理

import os
import io

import numpy as np
import cv2
import tensorflow as tf
import PIL

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(data_path_dict):
  image_input = data_path_dict["image_input"]
  label_input = data_path_dict["label_input"]
  tfrecord_output = data_path_dict["tfrecord_output"]
  tfrecord_name = data_path_dict["tfrecord_name"]

  if not os.path.exists(tfrecord_output):
    os.makedirs(tfrecord_output)
    
  output_filename = os.path.join(tfrecord_output,tfrecord_name)
  writer = tf.python_io.TFRecordWriter(output_filename)
  
  image_fnames = os.listdir(image_input)
  count = 0
  for idx, image_fname in enumerate(image_fnames):
    image_path = os.path.join(image_input, image_fname)
    label_path = os.path.join(label_input, image_fname).replace('image','label')
    print(image_path)
    print(label_path)
    
    with tf.gfile.GFile(image_path, 'rb') as fid:
      encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')

    with tf.gfile.GFile(label_path, 'rb') as fid:
      encoded_label = fid.read()
    encoded_label_io = io.BytesIO(encoded_label)
    label = PIL.Image.open(encoded_label_io)
    
    if image.size != label.size:
      raise ValueError('The size of image does not match with that of label.')

    width, height = image.size

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/encoded': bytes_feature(encoded_jpg),
        'label/encoded': bytes_feature(encoded_label),
      }))
    writer.write(tf_example.SerializeToString())
    count += 1
            
  print(count)
  writer.close()

  return          
            


if __name__ == '__main__':
  data_path_dict = {
    "image_input" : "../data_augmented/images/validation/",
    "label_input" : "../data_augmented/annotations/validation/",
    "tfrecord_output":"../data_tfrecords/",
    "tfrecord_name":"validation_data.tfrecord"
    }
  create_tfrecord(data_path_dict)









