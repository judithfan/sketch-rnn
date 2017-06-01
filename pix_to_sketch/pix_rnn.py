import numpy as np
import tensorflow as tf
import rnn
import databuilder
import data_manager

class pix2sketchrnn:
    def __init__(self, num_units, max_strokes, batch_size, learning_rate):
        self.learning_rate = learning_rate
        self.num_units = num_units #should be convo size
        self.num_hidden_units = 1024
        self.max_strokes = max_strokes #should be output
        self.convo_codes = tf.placeholder(tf.float64,
                            shape=(None, self.num_units),
                            name='convo_codes')
        self.labels = tf.placeholder(tf.float64,
                            shape=(None, self.max_strokes, 5),
                            name='labels')
        self.masks = tf.placeholder(tf.float64,
                            shape=(None, self.max_strokes),
                            name="masks")
        self.training = tf.placeholder(tf.bool,
                            shape = (),
                            name='training')
        self.batch_size = batch_size
        self.mew = .1
        self.rnn()

    def rnn(self):
        """
        Builds the entire RNN.
        """
        # Add linear transformation to get to hidden size
        W_xh = tf.get_variable('W_xh', [self.num_units, self.num_hidden_units],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        b_xh = tf.get_variable('b_xh', [self.num_hidden_units],
                               initializer=tf.zeros_initializer(), dtype=tf.float64)
        rnn_input = tf.nn.xw_plus_b(self.convo_codes, W_xh, b_xh)

        # Run RNN and run linear layer to fit to correct size.
        cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden_units)
        starting_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float64)
        self.outputs, final_rnn_state = tf.contrib.rnn.static_rnn(cell, 
                                    [rnn_input]*self.max_strokes,
                                    initial_state=starting_state,
                                    dtype=tf.float64)
        W_hy = tf.get_variable('W_hy', [self.num_hidden_units, 5],
                              initializer=tf.contrib.layers.xavier_initializer(),
                              dtype=tf.float64)
        self.preds = []
        for output in self.outputs:
            self.preds.append(tf.matmul(output, W_hy))
        
        # split the distance part of each 5-stroke and the p vector
        self.preds = tf.transpose(tf.stack(self.preds), perm=[1,0,2])
        self.p_preds = tf.nn.softmax(self.preds[:,:,2:])
        self.del_preds = self.preds[:,:,:2]

        if self.labels is not None:
            # If training, compute loss and optimize
            dist_loss = tf.reduce_sum(
                        tf.sqrt(
                        tf.reduce_sum(
                        tf.square(self.del_preds - self.labels[:, :, 0:2]),
                                  axis=2)) * self.masks, axis=1)
            pen_stroke_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                        labels = self.labels[:, :, 2:5],
                        logits = self.p_preds), axis= 1)
            self.loss = self.mew * dist_loss + pen_stroke_loss
            self.train_op = self.add_train_ops()
        # Compile 5 stroke and argmax on the p-vector
        maxed_tensor = tf.argmax(self.p_preds, axis=2)
        one_hot = tf.one_hot(maxed_tensor, depth=3,
                                 axis=-1, dtype=tf.float64)
        self.strokes = tf.concat([self.del_preds, one_hot], 2)

    def add_train_ops(self):
        """
        Adds the optimizer and minimizes the loss.
        """
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(self.loss)
        return train_op

    def train_batch(self, batch, masks, labels, sess, training = True):
        """
        Runs a training batch of data through the rnn.
        Returns the loss.
        """
        feed_dict = {
                self.convo_codes : batch,
                self.labels : labels,
                self.masks : masks,
                self.training: training
            }
        return sess.run([self.loss, self.train_op, self.strokes], feed_dict = feed_dict)

    def test_batch(self, batch, masks, labels, sess, training=False):
        """
        Runs a test batch through the rnn and returns the results
        in stroke 5 format.
        """
        feed_dict = {
                self.convo_codes : batch,
                self.labels : labels,
                self.masks : masks,
                self.training :training
                }
        return sess.run([self.loss, self.strokes], feed_dict = feed_dict)

    def evaluate(self, batch, masks, sess, training=False):
        feed_dict = {
                self.convo_codes : batch,
                self.masks: masks,
                self.training : training
                }
        return sess.run(self.strokes, feed_dict = feed_dict)


# TESTING CODE
#data = dataset.BadlyDrawnBunnies("data")
#conv_codes = np.load('conv_codes.npy')[()]
#all_sketches = np.load('all_sketches.npy')[()]
#print (conv_codes['airplane'][0].shape)
#sess = tf.Session()
#rnn = pix2sketchrnn(4096, 15, 100, 1e-9)
#init = tf.initialize_all_variables()
#sess.run(init)
#p_preds, del_preds = sess.run([rnn.p_preds, rnn.del_preds], {rnn.convo_codes:conv_codes['airplane'][0]})
#print (p_preds.shape)




