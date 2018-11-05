# coding=utf-8

import numpy as np

class Evaluation:

  def __init__(self, ins_dataprocess):
    self.ins_dataprocess = ins_dataprocess
    self.num_atoms = ins_dataprocess.num_atoms

  def get_all_evaluation(self, true_values, predicted_values):
    self.evaluate_force(true_values, predicted_values)

    #true_force, predicted_force = \
    #    true_values[:, 1: self.num_atoms * 3 + 1], \
    #    predicted_values[:, 1: self.num_atoms * 3 + 1]
    #print true_force.shape, predicted_force.shape
    #self.evaluate_force(true_force, predicted_force)

    #true_energy, predicted_energy = \
    #    true_values[:, 0], \
    #    predicted_values[:, 0]
    #self.evaluate_energy(true_energy, predicted_energy)

    #true_velocity, predicted_velocity = \
    #    true_values[:, self.num_atoms * 3 + 1: ], \
    #    predicted_values[:, self.num_atoms * 3 + 1: ]
    #print true_velocity.shape, predicted_velocity.shape
    #self.evaluate_velocity(true_velocity, predicted_velocity)

  def evaluate_force(self, true_force, predicted_force):
    mean_error_force = np.mean(np.abs(np.subtract(true_force, predicted_force)))
    print("  The mean error of force is: " + str(mean_error_force))

    label_min, label_max = \
        np.abs(self.ins_dataprocess.label_min), \
        self.ins_dataprocess.label_max
    if label_min > label_max:
      temp = label_max
    else:
      temp = label_min
    print("  The mean error rate of force is: " + \
        str(mean_error_force / temp * 100) + '%')

    true_force, predicted_force = \
        true_force.reshape(-1, 3), \
        predicted_force.reshape(-1, 3)
    #mean_error_force_each_direction = np.mean(
    #    np.abs(np.subtract(true_force, predicted_force)),
    #    axis = 0)
    #print("  The mean error of force in each direction is: " + \
    #    str(mean_error_force_each_direction))

  def evaluate_energy(self, true_energy, predicted_energy):
    mean_error_energy = np.mean(np.abs(np.subtract(true_energy, predicted_energy)))
    print("  The mean error of energy is: " + str(mean_error_energy))

  def evaluate_velocity(self, true_velocity, predicted_velocity):
    mean_error_velocity = np.mean(np.abs(np.subtract(true_velocity, predicted_velocity)))
    print("  The mean error of velocity is: " + str(mean_error_velocity))

    true_velocity, predicted_velocity = \
        true_velocity.reshape(-1, 3), \
        predicted_velocity.reshape(-1, 3)
    mean_error_velocity_each_direction = np.mean(
        np.abs(np.subtract(true_velocity, predicted_velocity)),
        axis = 0)
    print("  The mean error of velocity in each direction is: " + \
        str(mean_error_velocity_each_direction))
