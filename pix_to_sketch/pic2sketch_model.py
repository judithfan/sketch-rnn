import pickle
import tensorflow as tf
import numpy as np

from data_manager import DataManager
from pix_rnn import pix2sketchrnn


SAVE_FILENAME = 'best_pic2sketch_model'
PARAMS = {
        'epoch_size': 1000,
        'batch_size': 25, #100,
        'num_units': 4096,
        'max_strokes': 200, #get this from data managet
        'LR' : 1e-2,
        'mew' : .01
        }


class pic2sketch(object):
    def __init__(self, params, path):
        self.saver = tf.train.Saver()
        self.params = params
        self.epoch_size = params['epoch_size']
        self.batch_size = params['batch_size']
        self.manager = DataManager(path)
        self.sess = tf.Session()
        self.rnn = pix2sketchrnn(params['num_units'], params['max_strokes'],
                                self.batch_size, params['LR'])
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.train_set, self.dev_set, self.test_set = self.manager.get_full_dataset()

    def train(self, epochs=10, load=None, save=None):
        best_loss = -1
        best_model = self.rnn
        print('Running Epochs...')
        print('...')
        print('...')
        for i in range(epochs):
            loss = self.run_epoch()
            print("Epoch number: " + str(i) + ". Validation Loss: " + str(loss))
            if best_loss == -1 or loss < best_loss:
                best_loss = loss
                best_model = self.rnn
        if save is not None:
            self.saver.save(self.sess, save)

    def run_epoch(self):
        train_codes, train_sketches, train_mask = self.train_set
        print("Running epoch loop")
        for convs, padded_sketches, mask in self.manager.get_minibatches(
                                    [train_codes, train_sketches,
                                    train_mask], self.batch_size):
            if len(convs) == self.batch_size:
                loss, _, strokes = self.rnn.train_batch(convs, mask, padded_sketches, self.sess)
                print("Batch loss: " + str(sum(loss) / float(self.batch_size)))

        total_loss = 0
        for dev_c, dev_pad, dev_mask in self.manager.get_minibatches(
                                    [self.dev_set[0], self.dev_set[1],
                                    self.dev_set[2]], self.batch_size):
            if len(dev_c) == self.batch_size:
                loss, strokes = self.rnn.test_batch(dev_c, dev_mask, dev_pad, self.sess)
                total_loss += sum(loss) / float(len(loss))
                print("Strokes: " + str(strokes[0]))
                print("Labels: " + str(dev_pad[0]))
        return total_loss

    
    def test(self):
        test_batch, test_labels, test_masks = self.manager.get_minibatches(self.test_set, self.batch_size)
        loss, strokes = self.rnn.test_batch(test_batch, test_masks, test_labels, self.sess)
        print('Test loss is: ' + str(loss))

model = pic2sketch(PARAMS, '.')
print("WE DID IT!!")
model.train(save= SAVE_FILENAME)
