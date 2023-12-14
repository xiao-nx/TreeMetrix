from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import numpy as np

import unet_model
import preprocessing

import cv2
from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./data/test_images',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./data/inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./data/test_images/list.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--depth_multiplier', type=float, default=1.0,
                    help='Separable convolution channel depth multiplier.')

parser.add_argument('--channel_multiplier', type=float, default=1.0,
                    help='UNet channel depth multiplier.')

parser.add_argument('--use_separable_conv2d', action='store_true',
                    help='Whether use mobilenet seperateble convolution')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 3


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=unet_model.unet_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'channel_multiplier': FLAGS.channel_multiplier,
          'depth_multiplier': FLAGS.depth_multiplier,
          'use_separable_conv2d': FLAGS.use_separable_conv2d,
          'num_classes': _NUM_CLASSES,
      })

  examples = open(FLAGS.infer_data_list).readlines()
  image_files = [os.path.join(FLAGS.data_dir, filename.strip()) for filename in examples]

  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=pred_hooks)

  output_dir = FLAGS.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for pred_dict, image_path in zip(predictions, image_files):
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = image_basename + '_mask.png'
    path_to_output = os.path.join(output_dir, output_filename)

    print("generating:", path_to_output)
    mask = pred_dict['decoded_labels']
    cv2.imwrite(path_to_output, mask)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
