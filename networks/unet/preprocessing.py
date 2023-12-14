"""Utility functions for preprocessing data sets."""

from PIL import Image
import numpy as np
import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

# colour map
_LABEL_COLORS = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]

def decode_labels(mask, num_images=1, num_classes=3):
  """Decode batch of segmentation masks.

  Args:
    mask: result of inference after taking argmax.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).

  Returns:
    A batch with num_images RGB images of the same size as the input.
  """
  n, h, w, c = mask.shape
  assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                            % (n, num_images)
  outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
  for i in range(num_images):
    img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
    pixels = img.load()
    for j_, j in enumerate(mask[i, :, :, 0]):
      for k_, k in enumerate(j):
        if k < num_classes:
          pixels[k_, j_] = _LABEL_COLORS[k]
    outputs[i] = np.array(img)
  return outputs


def mean_image_addition(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
  """Adds the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] += means[i]
  return tf.concat(axis=2, values=channels)


def mean_image_subtraction(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def random_rescale_image_and_label(image, label, min_scale, max_scale):
  """Rescale an image and label with in target scale.

  Rescales an image and label within the range of target scale.

  Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[height, width, 1]`.
    min_scale: Min target scale.
    max_scale: Max target scale.

  Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
    If `labels` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, 1]`.
  """
  if min_scale <= 0:
    raise ValueError('\'min_scale\' must be greater than 0.')
  elif max_scale <= 0:
    raise ValueError('\'max_scale\' must be greater than 0.')
  elif min_scale >= max_scale:
    raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

  shape = tf.shape(image)
  height = tf.to_float(shape[0])
  width = tf.to_float(shape[1])
  scale = tf.random_uniform(
      [], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  image = tf.image.resize_images(image, [new_height, new_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
  # Since label classes are integers, nearest neighbor need to be used.
  label = tf.image.resize_images(label, [new_height, new_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return image, label

def eval_input_fn(image_filenames, label_filenames=None, batch_size=1):
  """An input function for evaluation and inference.

  Args:
    image_filenames: The file names for the inferred images.
    label_filenames: The file names for the grand truth labels.
    batch_size: The number of samples per batch. Need to be 1
        for the images of different sizes.

  Returns:
    A tuple of images and labels.
  """
  # Reads an image from a file, decodes it into a dense tensor
  def _parse_function(filename, is_label):
    if not is_label:
      image_filename, label_filename = filename, None
    else:
      image_filename, label_filename = filename

    image_string = tf.read_file(image_filename)
    image = tf.image.decode_image(image_string)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    image = mean_image_subtraction(image)

    if not is_label:
      return image
    else:
      label_string = tf.read_file(label_filename)
      label = tf.image.decode_image(label_string)
      label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
      label.set_shape([None, None, 1])

      return image, label

  if label_filenames is None:
    input_filenames = image_filenames
  else:
    input_filenames = (image_filenames, label_filenames)

  dataset = tf.data.Dataset.from_tensor_slices(input_filenames)
  if label_filenames is None:
    dataset = dataset.map(lambda x: _parse_function(x, False))
  else:
    dataset = dataset.map(lambda x, y: _parse_function((x, y), True))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()

  if label_filenames is None:
    images = iterator.get_next()
    labels = None
  else:
    images, labels = iterator.get_next()

  return images, labels
