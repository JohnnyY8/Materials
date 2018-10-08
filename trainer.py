# coding=utf-8

import random
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
    #print(self.x_train.shape, self.x_test.shape, self.y_train.shape, self.y_test.shape)

    with tf.Session() as sess:
      train_accu_old, train_accu_new, test_accu_best = 0.0, 0.0, 0.0
      num_epoches = 0

      saver = tf.train.Saver()
      sess.run(self.ins_model.init)

      while True:
        train_index = np.array(range(self.x_train.shape[0]))
        random.shuffle(train_index)
        print("No.%d epoch is starting..." % (num_epoches))
        for ind in xrange(0,
            self.x_train.shape[0],
            self.FLAGS.batch_size):

          batch_xs, batch_ys = \
              self.x_train[train_index[ind: ind + self.FLAGS.batch_size]], \
              self.y_train[train_index[ind: ind + self.FLAGS.batch_size]]

          train_loss, temp = sess.run(
                [self.ins_model.loss,
                 self.ins_model.train_step],
                feed_dict = {
                    self.ins_model.x_data: batch_xs,
                    self.ins_model.y_label: batch_ys,
                    self.ins_model.keep_prob: self.FLAGS.dropout_rate})
        print(train_loss)

        #if test_accu_new > test_accu_best:
        #  test_accu_best = test_accu_new
        #  save_path = saver.save(
        #      sess,
        #      os.path.join(self.FLAGS.path_save_model, "model.ckpt"))

        if num_epoches >= self.FLAGS.train_epoches:
          print("The training process is done...")
          print("The model saved in file:", save_path)
          break
        num_epoches += 1

    #return save_path
