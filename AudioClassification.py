"""Convolutional Neural Network Estimator for Urban Audio, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import ParseImages as pi
import os

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 50, 50, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 50, 50, 1]
  # Output Tensor Shape: [batch_size, 50, 50, 64]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 50, 50, 32]
  # Output Tensor Shape: [batch_size, 25, 25, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 25, 25, 32]
  # Output Tensor Shape: [batch_size, 25, 25, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 25, 25, 64]
  # Output Tensor Shape: [batch_size, 12, 12, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 12, 12, 64]
  # Output Tensor Shape: [batch_size, 12 * 12 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 12 * 12 * 64]
  # Output Tensor Shape: [batch_size, 2048]
  dense1 = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout2, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  image_data = pi.ReadImage()
  # Init numpy array that's 50*50 long for the image
  train_data = np.zeros(shape=(int(len(image_data)*0.85), 50*50), dtype=np.float32)
  train_labels = np.zeros(shape=(int(len(image_data)*0.85)), dtype=np.float32)
  eval_data = np.zeros(shape=(math.ceil(len(image_data)*0.15), 50*50), dtype=np.float32)
  eval_labels = np.zeros(shape=(math.ceil(len(image_data)*0.15)), dtype=np.float32)

  train_count = 0
  eval_count = 0
  train_data_length = int(len(image_data)*0.85)

  for name, data in image_data.items():
      if train_count < train_data_length:
          # add to the train data
          train_data[train_count, :] = np.float32(data[0]) # picture
          train_labels[train_count] = np.float32(data[1]) # label/truth
          train_count += 1
      else:
          # add to the eval data
          eval_data[eval_count, :] = np.float32(data[0]) # picture
          eval_labels[eval_count] = np.float32(data[1]) # label/truth
          eval_count += 1

  # Create the Estimator
  urban_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "urban_sound_model"))

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  urban_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = urban_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
