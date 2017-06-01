import numpy as np
import tensorflow as tf
import os
import svg_converter as svg
import cairosvg
import pickle
from scipy.misc import imread, imresize
import vgg16
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import time


# Only run this after running DataBuilder to generate .npy files
class DataManager():

    def __init__(self, path):

        def pad_sketches(all_sketches, max_lines):
            all_padded_sketches = {}
            for c in all_sketches:
                sketches, meta = all_sketches[c]
                padded_sketches = np.zeros((len(sketches), max_lines, 5), dtype=np.float64)
                mask = np.zeros((len(sketches), max_lines), dtype=np.bool)
                for i, sketch in enumerate(sketches):
                    padded_sketches[i, :sketch.shape[0], :] = sketch
                    mask[i, :sketch.shape[0]] = 1
                all_padded_sketches[c] = (padded_sketches, mask, meta)
            return all_padded_sketches

        def build_joint_data(conv_codes, padded_sketches):
            all_sketches = []
            all_images = []
            all_masks = []
            for c in padded_sketches:
                sketches, mask, meta = padded_sketches[c]
                images, meta_img = conv_codes[c]
                meta_img = {meta_img[k]:k for k in meta_img}
                for i, sketch in enumerate(sketches):
                    all_sketches.append(sketch)
                    all_masks.append(mask[i])
                    all_images.append(images[meta_img[meta[i]]])
            return all_images, all_sketches, all_masks

        def split_data(conv_codes, sketches, masks):
            all_idx = np.arange(0, len(masks))
            np.random.shuffle(all_idx)
            conv_codes = np.array(conv_codes)[all_idx]
            sketches = np.array(sketches)[all_idx]
            masks = np.array(masks)[all_idx]
            print (masks.shape)
            train_len = int(len(masks)*0.85)
            dev_len = int(len(masks)*0.05)
            test_len = int(len(masks*0.10))
            train = (conv_codes[:train_len], sketches[:train_len], masks[:train_len])
            dev = (conv_codes[train_len:train_len+dev_len], sketches[train_len:train_len+dev_len], masks[train_len:train_len+dev_len])
            test = (conv_codes[train_len+dev_len:], sketches[train_len+dev_len:], masks[train_len+dev_len:])
            print ("%d train examples" % (len(train[0])))
            print ("%d dev examples" % (len(dev[0])))
            print ("%d test examples" % (len(test[0])))
            return train, dev, test

        conv_codes = np.load('%s/conv_codes.npy'% (path))[()]
        all_sketches = np.load('%s/all_sketches.npy'% (path))[()]
        self.max_lines = max([max([sketch.shape[0] for sketch in all_sketches[c][0]]) for c in all_sketches])
        self.all_sketch_lines = [sketch.shape[0] for c in all_sketches for sketch in all_sketches[c][0]]
#        n, bins, patches = plt.hist(self.all_sketch_lines, 50, normed=1, facecolor='green', alpha=0.75)
#        plt.show()

        self.padded_sketches = pad_sketches(all_sketches, self.max_lines)
        self.conv_codes = conv_codes
        conv_codes, padded_sketches, masks = build_joint_data(conv_codes, self.padded_sketches)
        self.train_set, self.dev_set, self.test_set = split_data(conv_codes, padded_sketches, masks)

    def get_full_dataset(self):
        return self.train_set, self.dev_set, self.test_set

    def get_minibatches(self, data, minibatch_size, shuffle=True):
        """
        Iterates through the provided data one minibatch at at time. You can use this function to
        iterate through data in minibatches as follows:
            for inputs_minibatch in get_minibatches(inputs, minibatch_size):
                ...
        Or with multiple data sources:
            for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
                ...
        Args:
            data: there are two possible values:
                - a list or numpy array
                - a list where each element is either a list or numpy array
            minibatch_size: the maximum number of items in a minibatch
            shuffle: whether to randomize the order of returned data
        Returns:
            minibatches: the return value depends on data:
                - If data is a list/array it yields the next minibatch of data.
                - If data a list of lists/arrays it returns the next minibatch of each element in the
                  list. This can be used to iterate through multiple data sources
                  (e.g., features and labels) at the same time.
        """
        list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
        data_size = len(data[0]) if list_data else len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for minibatch_start in np.arange(0, data_size, minibatch_size):
            minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
            yield [minibatch(d, minibatch_indices) for d in data] if list_data \
                else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)

# Usage:
# dm = DataManager('.')
# train, dev, test = dm.get_full_dataset()

# train_codes, train_sketches, train_mask = train
# dev_codes, dev_sketches, dev_mask = dev
# test_codes, test_sketches, test_mask = test

# for batch_codes, batch_sketch, batch_mask in dm.get_minibatches([train_codes, train_sketches, train_mask], 50):
#     print(np.array(batch_codes).shape)
#     print(np.array(batch_sketch).shape)
#    print(np.array(batch_mask).shape)


