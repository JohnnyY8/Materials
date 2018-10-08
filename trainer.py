# coding=utf-8

import numpy as np
import tensorflow as tf

class Trainer:

  def __init__(self, FLAGS, ins_dataprocess, ins_model):
    self.FLAGS = FLAGS
    self.ins_dataprocess = ins_dataprocess
    self.ins_model = ins_model

  # Training and testing dnn model
  def train_dnn(self):
    self.x_train, self.x_test, self.y_train, self.y_test = \
        self.ins_dataprocess.split_data_2train_and_test()
    print(self.x_train.shape, self.x_test.shape, self.y_train.shape, self.y_test.shape)
