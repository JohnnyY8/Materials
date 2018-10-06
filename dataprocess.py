# coding=utf-8

import os
import numpy as np
import tensorflow as tf

class DataProcess:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS

    print("1.xx; 2.xx; 3.xx;")
    dic_elements = {1: "xx", 2: "xx", 3: "xx"}
    dic_num_atoms = {"xx": 32, "xx": 32, "xx": 32}
    choice = input("Please choose the element:")
    self.name_element = dic_elements[choice]
    self.num_atoms = dic_num_atoms[self.name_element]

  def load_data_and_label(self):
    self.load_data(), self.load_label()

  def load_data(self):
    print("1.position;")
    dic_data = {1: "position"}
    choice = input("Please choose the data:")
    self.data = dic_data[choice]

    path_data_file = os.path.join(
        self.FLAGS.path_all_data,
        self.name_element,
        self.data + ".dat")

  def load_label(self):
    print("1.force; 2.velocity;")
    dic_label = {1: "force", 2: "velocity"}
    choice = input("Please choose the label:")
    self.label = dic_label[choice]

    path_label_file = os.path.join(
        self.FLAGS.path_all_data,
        self.name_element,
        self.label + ".dat")
