# coding=utf-8

import numpy as np

class Evaluation:

  def __init__(self, ins_dataprocess):
    self.num_atoms = ins_dataprocess.num_atoms

  def get_all_evaluation(self, true_values, predicted_values):
    true_energy, predicted_energy = true_values[:, 0], predicted_values[:, 0]
    self.evaluate_energy(true_energy, predicted_energy)

    true_force, predicted_force = \
        true_values[:, 1: self.num_atoms * 3 + 1], \
        predicted_values[:, 1: self.num_atoms * 3 + 1]
    self.evaluate_force(true_force, predicted_force)

    true_velocity, predicted_velocity = \
        true_values[:, self.num_atoms * 3 + 1: ], \
        predicted_values[:, self.num_atoms * 3 + 1: ]
    self.evaluate_velocity(true_velocity, predicted_velocity)

  def evaluate_energy(self, true_energy, predicted_energy):
    res = np.mean(np.abs(np.subtract(true_energy, predicted_energy)))

    print("The mean error of energy is: " + str(res))

  def evaluate_force(self, true_force, predicted_force):
    res = np.mean(np.abs(np.subtract(true_force, predicted_force)))

    print("The mean error of force is: " + str(res))

  def evaluate_velocity(self, true_velocity, predicted_velocity):
    res = np.mean(np.abs(np.subtract(true_velocity, predicted_velocity)))

    print("The mean error of velocity is: " + str(res))
