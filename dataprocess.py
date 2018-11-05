# coding=utf-8

import os
import numpy as np
import tensorflow as tf

from sklearn.cross_validation import train_test_split

class DataProcess:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS

    print("1.Be; 2.C; 3.Li; 4.NaCl; 5.PbTe; 6.Si;")
    dic_elements = {1: "Be", 2: "C", 3: "Li", 4: "NaCl", 5: "PbTe", 6: "Si"}
    choice = input("Please choose the element: ")
    self.name_element = dic_elements[choice]

    dic_num_atoms = {"Be": 54, "C": 64, "Li": 32, "NaCl": 216, "PbTe": 216, "Si": 64}
    self.num_atoms = dic_num_atoms[self.name_element]
 
    dic_num_keyatoms = {"Be": 4, "C": 4, "Li": 12, "NaCl": 6, "PbTe": 24, "Si": 4}
    self.num_keyatoms = dic_num_keyatoms[self.name_element]

    dic_center_atoms = {"Be": 0, "C": 16, "Li": 29, "NaCl": 176, "PbTe": 41, "Si": 0}
    self.center_atom = dic_center_atoms[self.name_element]

  def load_data_and_label(self):
    self.load_single_data()
    #self.load_all_labels()
    self.load_single_label()

  def load_single_data(self):
    print("1.position;")
    dic_names_data = {1: "position"}
    choice = input("Please choose the data: ")
    self.name_data = dic_names_data[choice]

    path_data_file = os.path.join(
        self.FLAGS.path_all_data,
        self.name_element,
        self.name_data + ".npy")

    self.data = np.load(path_data_file).reshape(-1,
        self.num_atoms * self.FLAGS.num_directions)

    print("Loading " + self.name_data + " as single data is done.")

  def load_single_label(self):
    print("1.force;")
    dic_names_label = {1: "force"}
    choice = input("Please choose the label: ")
    self.name_label = dic_names_label[choice]

    path_label_file = os.path.join(
        self.FLAGS.path_all_data,
        self.name_element,
        self.name_label + ".npy")

    self.label = np.load(path_label_file).reshape(-1,
        self.num_atoms * self.FLAGS.num_directions)

    print("Loading " + self.name_label + " as single label is done.")

  def load_all_labels(self):
    print("1.energy; 2.force; 3.velocity;")
    list_labels = ["energy", "force", "velocity"]

    for ind, ele in enumerate(list_labels):
      path_label_file = os.path.join(
          self.FLAGS.path_all_data,
          self.name_element,
          ele + ".npy")
      if ind == 0:
        self.label = np.load(path_label_file)
      else:
        temp = np.load(path_label_file).reshape(-1,
            self.num_atoms * self.FLAGS.num_directions)
        self.label = np.hstack((self.label, temp))
    
    print("Loading all labels is done.")

  def split_data_2train_and_test(self):
    x_train, x_test, y_train, y_test = train_test_split(
        self.data,
        self.label,
        test_size = self.FLAGS.test_size,
        random_state = 24)

    return x_train, x_test, y_train, y_test

