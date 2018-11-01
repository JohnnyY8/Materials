# coding=utf-8

import os
import copy
import numpy as np
import tensorflow as tf

class SaliencyDetector:

  def __init__(self, FLAGS, ins_dataprocess, save_path):
    self.FLAGS = FLAGS
    self.ins_dataprocess = ins_dataprocess
    self.restore_model(save_path)


  def restore_model(self, save_path):
    self.sess = tf.Session()
    saver = tf.train.import_meta_graph(save_path + ".meta")
    graph = tf.get_default_graph()
    saver.restore(self.sess, save_path)

    self.x_data = graph.get_operation_by_name("x_data").outputs[0]
    self.y_label = graph.get_operation_by_name("y_label").outputs[0]
    self.keep_prob = graph.get_operation_by_name("dropout/keep_prob").outputs[0]
    self.z_output = graph.get_operation_by_name("output_layer/z_output").outputs[0]

  
  def cal_gradient(self):
    # Construct the grid
    length = self.ins_dataprocess.num_atoms * self.FLAGS.num_directions
    index_atom = int(input("Please input the number of atom: "))
    grid = self.ins_dataprocess.data[index_atom].reshape(1, length)

    # Calculate f(x).
    index_center = self.ins_dataprocess.center_atom * self.FLAGS.num_directions
    batch_x, batch_y = grid, np.zeros((1, length))
    f_all = self.run_model(batch_x, batch_y)[index_center - self.FLAGS.num_directions: index_center]
    
    # Calculate f(x + delta_x).
    delta = self.FLAGS.delta
    grid = grid.reshape(-1)
    batch_x = copy.deepcopy(grid)
    temp_z_x, temp_z_y, temp_z_z = \
        np.array([]), np.array([]), np.array([])

    for ind in xrange(length):
      #print("-----------" + str(ind) + "--------------")
      batch_x[ind] += delta
      batch_x = batch_x.reshape(1, length)
      f_delta_all = \
          self.run_model(batch_x, batch_y)[index_center - self.FLAGS.num_directions: index_center]
      temp_z_x = np.append(temp_z_x, f_delta_all[0])
      temp_z_y = np.append(temp_z_y, f_delta_all[1])
      temp_z_z = np.append(temp_z_z, f_delta_all[2])
      batch_x = batch_x.reshape(-1)
      batch_x[ind] = grid[ind]

    # Calculate gradient on x, y, z axises.
    f_delta_x = temp_z_x.reshape(-1, self.FLAGS.num_directions)[:, 0]
    f_delta_y = temp_z_y.reshape(-1, self.FLAGS.num_directions)[:, 1]
    f_delta_z = temp_z_z.reshape(-1, self.FLAGS.num_directions)[:, 2]
    
    gradient_x_x = (f_delta_x - f_all[0]) / delta
    print("The saliency on x aiex: " + str(self.select_max_gradient(gradient_x_x)))
    gradient_y_y = (f_delta_y - f_all[1]) / delta
    print("The saliency on y aiex: " + str(self.select_max_gradient(gradient_y_y)))
    gradient_z_z = (f_delta_z - f_all[2]) / delta
    print("The saliency on z aiex: " + str(self.select_max_gradient(gradient_z_z)))


  def run_model(self, batch_x, batch_y):
    feed_data = {self.x_data: batch_x, self.y_label: batch_y, self.keep_prob: 1.0}
    temp_z = self.sess.run(self.z_output, feed_dict = feed_data)

    return temp_z.reshape(-1)


  def select_max_gradient(self, gradient):
    gradient_abs = np.abs(gradient)
    gradient_sort = np.argsort(gradient_abs) + 1
    return gradient_sort[-(self.ins_dataprocess.num_keyatoms + 1): ][::-1]
