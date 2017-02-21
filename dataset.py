
import os
import glob
import json
import time
import math
import copy
import datetime
import warnings
import concurrent.futures

import bcolz
import numpy as np
import pandas as pd

import skimage
from skimage import transform
from skimage import img_as_ubyte

from keras.utils import np_utils
from PIL import Image, ImageDraw

import utils
import config

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

warnings.filterwarnings("ignore")
np.random.seed(config.random_state)

class ImageAugmenter(object):
    def __init__(self, prefix, class_labels):
        self.prefix = prefix
        self.class_labels = class_labels
        self.images = self.get_images()
        self.bboxes = {}
        self.img_w = 0
        self.img_h = 0
        self.factor = 0
        self.rotation_range = 0
        self.translation_range = 0
        self.x_scale_range = 0
        self.y_scale_range = 0


    def set_bboxes(self, path):
        self.bboxes = {}
        for annotation_path in glob.glob(os.path.join(path, '**', '*.json'),
                                         recursive=True):
            self.bboxes.update(utils.read_bbox_json(annotation_path))


    def get_images(self):
        images = []
        d = os.getcwd()
        os.chdir(self.prefix)
        images = glob.glob(os.path.join('**', '*.jpg'), recursive=True)
        os.chdir(d)
        return images


    def get_bbox(self, path, label):
        bbox = [0, 0, 0, 0]
        if path in self.bboxes and label in self.bboxes[path]:
            # get random bbox
            idx = np.random.randint(0, len(self.bboxes[path][label]))
            bbox = self.bboxes[path][label][idx]
        return np.asarray(bbox, dtype=np.float32)


    def get_label(self, path, categorical=True):
        label = None
        if self.class_labels:
            if categorical:
                label = os.path.dirname(path)
                if label != '':
                    label = self.class_labels.index(label)
                    label = np_utils.to_categorical([label],
                                                    len(self.class_labels))
                    label = label[0]
            else:
                label = os.path.dirname(path)
        return label


    def run(self, save=True):
        print('Augmenting \'{}\'. #images: {}, #bboxes: {}'.
              format(self.prefix,len(self.images), len(self.bboxes)))
        res_images = None
        res_bboxes = None
        res_labels = None
        res_ids = None
        init = False

        with concurrent.futures.ThreadPoolExecutor() as executor:
            t0 = time.time()

            futures = {
                executor.submit(self.read_image, img_path):
                img_path for img_path in np.random.permutation(self.images)
            }
            for f in concurrent.futures.as_completed(futures):
                img_path = futures[f]
                img, bbox = f.result()
                label = self.get_label(img_path)

                if not init:
                    res_images = bcolz.carray([img], rootdir=self.prefix +
                                              '_images.bc', mode='w')
                    res_bboxes = bcolz.carray([bbox], rootdir=self.prefix +
                                              '_bboxes.bc', mode='w')
                    res_labels = bcolz.carray([label], rootdir=self.prefix +
                                              '_labels.bc', mode='w')
                    res_ids = bcolz.carray([img_path], rootdir=self.prefix +
                                           '_ids.bc', mode='w')
                    init = True
                else:
                    res_images.append(img)
                    res_bboxes.append(bbox)
                    res_labels.append(label)
                    res_ids.append(img_path)
            t1 = time.time()
            print('Finished reading {} images in {:.2f} s'.
                  format(len(self.images), t1 - t0))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(self.factor):
                t0 = time.time()
                futures = {
                    executor.submit(self.augment, img_path):
                    img_path for img_path in np.random.permutation(self.images)
                }

                for f in concurrent.futures.as_completed(futures):
                    img_path = futures[f]
                    img, bbox = f.result()
                    res_images.append(img)
                    res_bboxes.append(bbox)
                    res_labels.append(self.get_label(img_path))
                    res_ids.append(img_path)

                t1 = time.time()
                print('Finished augmenting {}/{}: {} images in {:.2f} s'.
                      format(i+1, self.factor, len(self.images), t1 - t0))

                res_images.flush()
        res_bboxes.flush()
        res_labels.flush()
        res_ids.flush()
        print('Done')
        return res_images, res_bboxes, res_labels, res_ids


    def read_image(self, img_path):
        img = Image.open(os.path.join(self.prefix, img_path))
        x_scale = float(self.img_w) / float(img.width)
        y_scale = float(self.img_h) / float(img.height)
        img = img.resize((config.img_w, config.img_h), Image.ANTIALIAS)
        bbox = self.get_bbox(img_path, self.get_label(img_path, False))
        bbox[[0,2]] *= x_scale
        bbox[[1,3]] *= y_scale
        return np.asarray(img, dtype=np.uint8), bbox


    def augment(self, img_path):
        img, bbox = self.read_image(img_path)
        img, bbox, _ = box_zoom_rotate_translate(img, bbox,
                                                 self.x_scale_range,
                                                 self.y_scale_range,
                                                 self.rotation_range,
                                                 self.translation_range)
        return img, bbox


def read_relabels(path):
    # https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/28150/unified-effort-to-relabel-the-training-set
    relabels = {}
    with open(path) as f:
        for line in f:
            cols = line.split()
            src = "{}/{}/{}.jpg".format(config.training_images_path, cols[1], cols[0])
            relabels[src] = cols[2]
    return relabels


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


def translation(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def scale(sx, sy):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0,  1]])


def box_zoom_rotate_translate(img, bb, x_scale_range,
                              y_scale_range, rotation_range,
                              translation_range, mode='edge'):
    """Performs zoom of a Numpy image tensor.
    # Arguments
        img: Input image tensor (w, h, c).
        bbox: Bounding box tuple/array [x, y, w, h]
        x_scale_range: [1, sx]: x scale range.
        y_scale_range: [1, sy]: y scale range.
        rotation_range: Rotation range.
        translation_range: Translation range.
        mode: 
    # Returns
        Zoomed, rotated and translated numpy image tensor (w, h, c).
        New axis aligned bounding box [x, y, w, h].
        Transformed original bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
    # Raises
        None
    """

    sx = sy = np.random.uniform(1, (x_scale_range + y_scale_range) / 2.0)
    # sx = np.random.uniform(1, x_scale_range)
    # sy = np.random.uniform(1, y_scale_range)
    theta = np.random.uniform(-rotation_range, rotation_range)
    rtx = np.random.uniform(-translation_range, translation_range)
    rty = np.random.uniform(-translation_range, translation_range)

    if bb[2] == 0 and bb[3] == 0:
        zcx = (img.shape[0] / 2.0)
        zcy = (img.shape[1] / 2.0)
        rtx = zcx - rtx
        rty = zcy - rty
        box_tl = [0, 0]
        box_br = [0, 0]
        box = [0, 0, 0, 0]
    else:
        # zoom center
        zcx = (bb[0] + bb[2] / 2.0)
        zcy = (bb[1] + bb[3] / 2.0)

        # box top left
        box_tl = [bb[0], bb[1], 1]
        # box bottom right
        box_br = [bb[0] + bb[2], bb[1] + bb[3], 1]

        # transformation matrices
        tm = translation(-zcx, -zcy)
        sm = scale(sx, sy)
        rm = rotation(-theta)

        # rotate and zoom around the center of bb
        t = np.dot(rm, np.dot(sm, tm))

        # calculate zoomed and rotated bounding box
        v = np.array([box_br[0] - box_tl[0], 0, 1])

        box_tl = np.dot(t, box_tl)
        box_br = np.dot(t, box_br)
        v = np.dot(rm, np.dot(sm, v))

        box = np.array([box_tl, box_tl + v, box_br, box_br - v])

        # calculate min and max translation so that the final axis aligned
        # box remains inside the image
        tl_x = np.min([p[0] for p in box])
        tl_y = np.min([p[1] for p in box])
        br_x = np.max([p[0] for p in box])
        br_y = np.max([p[1] for p in box])

        min_translation = -1 * np.array([tl_x, tl_y])
        max_translation = img.shape[:2] - np.array([br_x, br_y])

        # get random translation between min and max
        rtx += zcx
        rty += zcy
        rtx = np.max([min_translation[0], rtx])
        rtx = np.min([max_translation[0], rtx])

        rty = np.max([min_translation[1], rty])
        rty = np.min([max_translation[1], rty])

        t2 = translation(rtx, rty)

        # calculate final axis aligned bounding box
        box_tl = np.dot(t2, box_tl)
        box_br = np.dot(t2, box_br)

        box = np.array([box_tl, box_tl + v, box_br, box_br - v])
        box_tl[0] = np.min([p[0] for p in box])
        box_tl[1] = np.min([p[1] for p in box])
        box_br[0] = np.max([p[0] for p in box])
        box_br[1] = np.max([p[1] for p in box])

    # transform the image
    tc = transform.SimilarityTransform(matrix=translation(zcx, zcy))
    tz = transform.SimilarityTransform(matrix=scale(1.0 / sx, 1.0 / sy))
    tr = transform.SimilarityTransform(matrix=rotation(theta))
    tu = transform.SimilarityTransform(matrix=translation(-rtx, -rty))

    img = img_as_ubyte(transform.warp(img, tu + tr + tz + tc, mode=mode)) 
    return img, [box_tl[0], box_tl[1], box_br[0] - box_tl[0], box_br[1] - box_tl[1]], box

augmenter = ImageAugmenter('train', classes)
augmenter.set_bboxes(config.bbox_annotations_path)
augmenter.img_w = config.img_w
augmenter.img_h = config.img_h
augmenter.factor = config.augmentation_factor
augmenter.rotation_range = config.augmentation_rotation_range
augmenter.translation_range = config.augmentation_translation_range
augmenter.x_scale_range = config.augmentation_x_scale_range
augmenter.y_scale_range = config.augmentation_y_scale_range

train_images, train_bboxes, train_labels, train_ids = augmenter.run(save=True)
print('Train images shape:', train_images.shape)
print('Train boxes shape:', train_bboxes.shape)
print('Train labels shape:', train_labels.shape)
print('Train ids shape:', train_ids.shape)
del train_images
del train_bboxes
del train_labels
del train_ids

augmenter = ImageAugmenter('test_stg1', classes)
augmenter.img_w = config.img_w
augmenter.img_h = config.img_h
augmenter.factor = config.augmentation_factor
augmenter.rotation_range = config.augmentation_rotation_range
augmenter.translation_range = 0 # config.augmentation_translation_range
augmenter.x_scale_range = config.augmentation_x_scale_range
augmenter.y_scale_range = config.augmentation_y_scale_range

test_images, test_bboxes, test_labels, test_ids = augmenter.run(save=True)
print('Test images shape: ', test_images.shape)
print('Test bboxes shape: ', test_bboxes.shape)
print('Test labels shape: ', test_labels.shape)
print('Test ids shape: ', test_ids.shape)
