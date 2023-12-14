from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import arg_scope

import preprocessing

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4

def unet_generator(num_classes, channel_multiplier, depth_multiplier, use_separable_conv2d=False):
  def _unet_conv_layers(feature, num_outputs, scope):
    print('Use separable conv2d={}, channel_multiplier={}, depth_multiplier={}'.format(use_separable_conv2d, channel_multiplier, depth_multiplier))
    net = feature
    for i in range(2):
      if use_separable_conv2d:
        net = layers.separable_conv2d(net, num_outputs, [3, 3], depth_multiplier, padding='SAME', scope=scope + '_separable_conv{}'.format(i+1))
      else:
        net = layers.conv2d(net, num_outputs, [3, 3], padding='SAME', scope=scope + '_conv{}'.format(i+1))
    return net

  def _unet_deconv_layes(sfeature, bfeature, scope, is_deconv=True):
    shape = tf.shape(bfeature)[1:3]
    num_outputs = bfeature.get_shape().as_list()[3]
    if is_deconv:
      sfeature = layers.conv2d_transpose(sfeature, num_outputs, [2, 2], stride=2, scope=scope + "_deconv")
    else:
      sfeature = tf.image.resize_bilinear(sfeature, shape, name=scope + "_resize")
      sfeature = layers.conv2d(sfeature, num_outputs, [1, 1], scope=scope + '_pointwise')
    net = tf.concat([sfeature, bfeature], axis=3, name=scope + "_concat")
    net = _unet_conv_layers(net, num_outputs=num_outputs, scope=scope)
    return net

  def model(inputs, is_training):
    with tf.variable_scope("UNet"):
      with arg_scope([layers.conv2d, layers.conv2d_transpose, layers.separable_conv2d], biases_initializer=tf.zeros_initializer()):
        with tf.variable_scope("Encode"):
          # 1
          net1 = _unet_conv_layers(inputs, 64 * channel_multiplier, scope='block1')
          # 1/2
          net2 = layers.max_pool2d(net1, [2, 2], scope='pool2')
          net2 = _unet_conv_layers(net2, 128 * channel_multiplier, scope='block2')
          # 1/4
          net3 = layers.max_pool2d(net2, [2, 2], scope='pool3')
          net3 = _unet_conv_layers(net3, 256 * channel_multiplier, scope='block3')
          # 1/8
          net4 = layers.max_pool2d(net3, [2, 2], scope='pool4')
          net4 = _unet_conv_layers(net4, 512 * channel_multiplier, scope='block4')
          # 1/16
          net5 = layers.max_pool2d(net4, [2, 2], scope='pool5')
          net5 = _unet_conv_layers(net5, 1024 * channel_multiplier, scope='block5')
        with tf.variable_scope("Decode"):
          # 1/8
          net6 = _unet_deconv_layes(net5, net4, scope='block1')
          # 1/4
          net7 = _unet_deconv_layes(net6, net3, scope='block2')
          # 1/2
          net8 = _unet_deconv_layes(net7, net2, scope='block3')
          # 1
          net9 = _unet_deconv_layes(net8, net1, scope='block4')
      logits = layers.conv2d(net9, num_classes, [3, 3], activation_fn=None, biases_initializer=None, scope='logits')
    return logits

  return model


def unet_model_fn(features, labels, mode, params):
  if isinstance(features, dict):
    features = features['feature']

  images = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, features),
      tf.uint8)

  network = unet_generator(params['num_classes'], params['channel_multiplier'],
                           params['depth_multiplier'], params['use_separable_conv2d'])

  logits = network(features, mode == tf.estimator.ModeKeys.TRAIN)

  pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

  pred_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                   [pred_classes, params['batch_size'], params['num_classes']],
                                   tf.uint8)

  predictions = {
      'classes': pred_classes,
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
      'decoded_labels': pred_decoded_labels
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
    predictions_without_decoded_labels = predictions.copy()
    del predictions_without_decoded_labels['decoded_labels']

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'preds': tf.estimator.export.PredictOutput(
                predictions_without_decoded_labels)
        })

  gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                 [labels, params['batch_size'], params['num_classes']], tf.uint8)

  labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.

  logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
  labels_flat = tf.reshape(labels, [-1, ])

  valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

  preds_flat = tf.reshape(pred_classes, [-1, ])
  valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
  confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

  predictions['valid_preds'] = valid_preds
  predictions['valid_labels'] = valid_labels
  predictions['confusion_matrix'] = confusion_matrix

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=valid_logits, labels=valid_labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  train_var_list = [v for v in tf.trainable_variables()]
  # Add weight decay to the loss.
  with tf.variable_scope("total_loss"):
    loss = cross_entropy + params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])
  # loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf.summary.image('images',
                     tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
                     max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

    global_step = tf.train.get_or_create_global_step()

    # poly learning rate
    learning_rate = tf.train.polynomial_decay(
        params['initial_learning_rate'],
        tf.cast(global_step, tf.int32) + params['initial_global_step'],
        params['max_iter'], params['end_learning_rate'], power=params['power'])

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['momentum'])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      valid_labels, valid_preds)
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
  metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_px_accuracy')
  tf.summary.scalar('train_px_accuracy', accuracy[1])

  def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['num_classes']):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result

  train_mean_iou = compute_mean_iou(mean_iou[1])

  tf.identity(train_mean_iou, name='train_mean_iou')
  tf.summary.scalar('train_mean_iou', train_mean_iou)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)
