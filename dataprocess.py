# coding=utf-8

import os
import numpy as np
import tensorflow as tf

class DataProcess:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS

    print("1.Be; 2.Li; 3.Si;")
    dic_elements = {1: "Be", 2: "Li", 3: "Si"}
    choice = input("Please choose the element:")
    self.name_element = dic_elements[choice]

  def load_data_and_label(self):
    self.load_data(), self.load_label()

  def load_data(self):
    print("1.position;")
    dic_data = {1: "position"}
    choice = input("Please choose the data:")
    self.data = dic_data[choice]

    dic_num_atoms = {"Be": 54, "Li": 32, "Si": 64}
    self.num_atoms = dic_num_atoms[self.name_element]

    path_data_file = os.path.join(
        self.FLAGS.path_all_data,
        self.name_element,
        self.data + ".dat")

  def load_label(self):
    print("1.force; 2.velocity; 3.energy;")
    dic_label = {1: "force", 2: "velocity", 3: "energy"}
    choice = input("Please choose the label:")
    self.label = dic_label[choice]

    dic_num_properties = {
        "force": self.num_atoms,
        "velocity": self.num_atoms,
        "energy": 1}
    self.num_properties = dic_num_properties[self.label]

    path_label_file = os.path.join(
        self.FLAGS.path_all_data,
        self.name_element,
        self.label + ".dat")
