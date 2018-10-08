# coding=utf-8

import numpy as np
import tensorflow as tf

from commonfunction import *

class Model(CommonFunction):

  def __init__(self, FLAGS, num_neurons):
    self.FLAGS = FLAGS
    self.num_neurons = np.array(num_neurons)
    self.num_layers = self.num_neurons.shape[0]

  def build_model_graph(self):
    with tf.name_scope("dropout"):
      self.keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
      #tf.summary.scalar("dropout_keep_probability", self.keep_prob)

    for ind, ele in enumerate(self.num_neurons):
      if ind == 0:  # Input layer
        self.x_data = tf.placeholder(tf.float32, [None, ele], name = "x_data")

      elif ind == self.num_layers - 1:  # Output layer
        self.y_label = tf.placeholder(tf.float32, [None, ele], name = "y_label")

        name_variable_scope = "output_layer"
        with tf.variable_scope(name_variable_scope):
          name_w, name_b = "w_output", "b_output"
          name_z, name_h = "z_output", "h_output"

          w_output = self.init_weight_variable(
              name_w,
              [self.num_neurons[ind - 1], ele])
          #self.variable_summaries(w_hidden)

          b_output = self.init_bias_variable(name_b, [ele])
          #self.variable_summaries(bHidden)

          self.z_output = tf.add(tf.matmul(h_hidden, w_output),
              b_output, name = name_z)
          #self.variable_summaries(self.h_output)

          #self.h_output = tf.nn.softmax(self.h_output, name = name_h)
          #self.variable_summaries(self.y_output)

      else:  # Hidden layers
        name_variable_scope = "hidden" + str(ind) + "_layer"

        with tf.variable_scope(name_variable_scope):
          name_w, name_b = "w_hidden" + str(ind), "b_hidden" + str(ind)
          name_z, name_h = "z_hidden" + str(ind), "h_hidden" + str(ind)

          w_hidden = self.init_weight_variable(name_w,
              [self.num_neurons[ind - 1], ele])
          #self.variable_summaries(wHidden)

          b_hidden = self.init_bias_variable(name_b, [ele])
          #self.variable_summaries(bHidden)

          if ind == 1:
            z_hidden = tf.add(tf.matmul(self.x_data, w_hidden),
                b_hidden, name = name_z)
            h_hidden = tf.nn.relu(z_hidden, name = name_h)
          else:
            z_hidden = tf.add(tf.matmul(h_hidden, w_hidden),
                b_hidden, name = name_z)
            h_hidden = tf.nn.relu(z_hidden, name = name_h)
          #self.variable_summaries(preAct)

          #if ind == self.num_layers - 2:
          #  h_idden = tf.nn.dropout(h_hidden, self.keep_prob)
          #self.variable_summaries(hHidden)

    self.get_least_squares_method()

  def get_least_squares_method(self):
    name_variable_scope = "loss_layer"

    with tf.name_scope(name_variable_scope):
      self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.y_label - self.z_output), axis = 1))

    self.get_optimizer()

  def get_optimizer(self):
    self.train_step = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(self.loss)







