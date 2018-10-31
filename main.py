# coding=utf-8

import os
import numpy as np
import tensorflow as tf

from dataprocess import *
from trainer import *
from model import *
from evaluation import *
from saliencydetector import *

flags = tf.app.flags

flags.DEFINE_string(
    "gpu_id",
    "0",
    "Which gpu is assigned.")

flags.DEFINE_string(
    "path_all_files",
    "./files",
    "The path for all files.")

flags.DEFINE_string(
    "path_all_data",
    "./files/data",
    "The path for all data files.")

flags.DEFINE_string(
    "path_save_model",
    "./files/trained_model",
    "The path for saving model.")

flags.DEFINE_float(
    "test_size",
     1e-1,
     "The rate for test data.")

flags.DEFINE_float(
     "learning_rate",
     1e-3,
     "The learning rate for training.")

flags.DEFINE_float(
     "delta",
     1e-4,
     "The offset for each point.")

flags.DEFINE_float(
     "dropout_rate",
     0.5,
     "The dropout rate for model.")

flags.DEFINE_float(
     "threshold_test",
     0.5,
     "The threshold for test data.")

flags.DEFINE_integer(
     "batch_size",
     150,
     "How many samples are trained in each iteration.")

flags.DEFINE_integer(
     "train_epoches",
     1,
     "How many times training through all train data.")

flags.DEFINE_integer(
     "num_directions",
     3,
     "How many directions.")

FLAGS = flags.FLAGS

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
  ins_dataprocess = DataProcess(FLAGS)
  ins_dataprocess.load_data_and_label()

  print("The shapes of data and label are: " + \
      str(ins_dataprocess.data.shape) + ", " + \
      str(ins_dataprocess.label.shape)) + '.'
  #print(ins_dataprocess.data[0][84: 87])
  raw_input("Press Enter to continue.")

  num_neurons = [ins_dataprocess.num_atoms * FLAGS.num_directions,
      ins_dataprocess.num_atoms, 
      22, 16, 12, 9,
      ins_dataprocess.num_atoms * FLAGS.num_directions]
  ins_model = Model(FLAGS, num_neurons)
  ins_model.build_model_graph()

  ins_evaluation = Evaluation(ins_dataprocess)
  ins_trainer = Trainer(FLAGS, ins_dataprocess, ins_model, ins_evaluation)
  save_path = ins_trainer.train_dnn()

  ins_saliencydetector = SaliencyDetector(FLAGS, ins_dataprocess, save_path)
  ins_saliencydetector.cal_gradient()
