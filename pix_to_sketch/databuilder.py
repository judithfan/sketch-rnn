import numpy as np
import tensorflow as tf
import os
import svg_converter as svg
#import cairosvg
import pickle
from scipy.misc import imread, imresize
import vgg16


# Run this code to build the data files appropriately
class DataBuilder():

    def __init__(self, path):

        def parseSketches(filepath):
            all_classes = {}
            all_strokes = []
            for p in os.listdir(filepath):
                if p == ".DS_Store": continue
                all_ex = []
                lookup_dict = {}
                counter = 0
                with open("%s/%s/invalid.txt" % (filepath, p)) as f:
                    invalid = set([line[:-1] for line in f.readlines()])
                total = len(os.listdir("%s/%s" % (filepath, p)))
                for ex in os.listdir("%s/%s" % (filepath, p)):
                    if ex == ".DS_Store" or ex == "checked.txt" or ex == "invalid.txt": continue
                    if ex[:-4] in invalid:
                        print ("this one's bs bruh")
                        total -= 1
                        continue
                    print ("%s/%s/%s" % (filepath, p, ex))
                    strokes = svg.svg_to_stroke5("%s/%s/%s" % (filepath, p, ex))
                    if strokes == None:
                        total -= 1
                        print ("Invalid")
                        continue
                    if (strokes.shape[0] > 200): continue
                    all_strokes.append(strokes.shape[0])
                    all_ex.append(strokes)
                    lookup_dict[counter] = ex[:ex.find("-")]
                    print ("%d / %d" % (counter, total))
                    counter += 1
                all_classes[p] = (all_ex, lookup_dict)
                break
            print (all_classes)
            np.save('all_sketches.npy', all_classes)
            print (max(all_strokes))
            return all_classes, max(all_strokes)

        def parsePhotos(filepath):
            all_classes = {}
            for c in os.listdir(filepath):
                all_imgs = []
                lookup_dict = {}
                for i, img_name in enumerate(os.listdir("%s/%s" % (filepath, c))):
                    print (img_name)
                    if img_name == ".DS_Store": continue
                    img = imread("%s/%s/%s" % (filepath, c, img_name))
                    img = imresize(img, (224, 224))
                    all_imgs.append(img)
                    lookup_dict[i] = img_name[:-4]
                print("Length of all imgs:",len(all_imgs))
                codes = vgg16.generate_conv_codes(all_imgs)
                all_classes[c] = (codes, lookup_dict)
                break
            print ("Pickling full object")
            np.save('conv_codes.npy', all_classes)
            return all_classes


        self.files = parsePhotos("%s/photos/tx_000000000000" % (path))
        self.sketches = parseSketches("%s/sketches" % (path))

if __name__ == '__main__':
    build = DataBuilder('data')


