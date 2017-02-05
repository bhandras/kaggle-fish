#!/usr/bin/env python3

import os
import glob
import time
import pickle
import argparse
import datetime

import utils
import config

import numpy as np
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam, Nadam, Adamax, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Input, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, BatchNormalization

from sklearn.metrics import log_loss
from keras.layers.advanced_activations import PReLU
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


def read_relabels(path):
    # https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/28150/unified-effort-to-relabel-the-training-set
    relabels = {}
    with open(path) as f:
        for line in f:
            cols = line.split()
            src = "{}/{}/{}.jpg".format(config.training_images_path, cols[1], cols[0])
            relabels[src] = cols[2]
    return relabels


def read_bbox_annotations(path):
    boxes = {}
    for c in classes:
        path = os.path.join(path, c + '.json')
        if os.path.isfile(path):
            class_boxes = utils.read_bbox_json(path)
            boxes.update(class_boxes)
    return boxes


def preprocess_img_data(img_arr):
    preprocess_input(img_arr)


def read_training_images(path, boxes, relabels):
    X_train = []
    y_train = []
    y_train_box = []

    t0 = time.time()
    print('Reading training images...')

    for c in classes:
        images = glob.glob(os.path.join(path, c, '*.jpg'))
        class_index = classes.index(c)
        print('Loading class: {}'.format(c))

        for img_path in images:
            # print('Reading: ', img_path)
            img_name = os.path.basename(img_path)

            img = image.load_img(img_path)
            x_scale = float(config.img_w) / float(img.width)
            y_scale = float(config.img_h) / float(img.height)
            img = img.resize((config.img_w, config.img_h))

            max_box = [0, 0, 0, 0]
            # get the largest bbox
            if c in boxes and img_name in boxes[c]:
                img_boxes = boxes[c][img_name]
                max_area = 0
                for box in img_boxes:
                    box_area = box[2] * box[3]
                    if box_area > max_area:
                        max_area = box_area
                        max_box = box

            max_box[0] *= x_scale
            max_box[1] *= y_scale
            max_box[2] *= x_scale
            max_box[3] *= y_scale

            add_img = True
            img_class = class_index
            if img_path in relabels:
                if relabels[img_path] == 'revise':
                    add_img = False
                    print('Image omitted: ', img_path)
                else:
                    print('Label revised: ', img_path, relabels[img_path])
                    img_class = classes.index(relabels[img_path])

            if add_img:
                img = image.img_to_array(img)
                X_train.append(img)
                y_train.append(img_class)
                y_train_box.append(max_box)

    X_train = np.array(X_train)
    preprocess_img_data(X_train)
    y_train = np.array(y_train, dtype=np.uint8)
    y_train = np_utils.to_categorical(y_train, 8)
    y_train_box = np.array(y_train_box, dtype=np.float32)

    t1 = time.time()
    print('Reading finished: {} seconds'.format(round(t1 - t0, 2)))
    print('Training data shape:', X_train.shape)
    return X_train, y_train, y_train_box


def read_testing_images(path):
    X_test = []
    Id_test = []

    print('Reading testing data...')
    t0 = time.time()
    images = glob.glob(os.path.join(path, '*.jpg'))

    for img_path in images:
        img = image.load_img(img_path, target_size=(config.img_h, config.img_w))
        X_test.append(image.img_to_array(img))
        Id_test.append(os.path.basename(img_path))

    X_test = np.array(X_test)
    Id_test = np.array(Id_test)
    X_test = preprocess_input(X_test)

    t1 = time.time()
    print('Reading finished: {} seconds'.format(round(t1 - t0, 2))) 
    print('Test data shape:', X_test.shape)
    return X_test, Id_test


def create_vgg(w, h):
    model = VGG16(include_top=False, weights='imagenet',
                  input_tensor=Input(shape=(h, w, 3)))
    for layer in model.layers:
        layer.trainable = False
    return model


def predict(X, model, batch_size):
    y_pred = []
    for batch in range(0, len(X), batch_size):
        y_pred.extend(model.predict(X[batch:batch + batch_size]))
    return np.array(y_pred)


def run_test(models, X_test, batch_size):
    y_class = []
    y_tl = []
    y_wh = []

    for i in range(len(models)):
        print('Testing model # {}/{}'.format(i+1, len(models)))
        y = models[i].predict(X_test, batch_size=batch_size, verbose=1)
        y_class.append(y[0])
        y_tl.append(y[1])
        y_wh.append(y[2])

    return np.mean(y_class, axis=0), np.mean(y_tl, axis=0), np.mean(y_wh, axis=0)


def save_test_result(y_test, id_test, id_name, info_string, index):
    result = pd.DataFrame(y_test, columns=classes)
    result.loc[:, id_name] = pd.Series(id_test, index=result.index)

    now = datetime.datetime.now()
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    filename = os.path.join('cache', info_string + '_' + str(index) + '_' +
                            str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv')
    print('Saving test result to: ', filename)
    result.to_csv(filename, index=False)


def save_model(model, history, info_string, index):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    name = os.path.join('cache', info_string + '_' + str(index))
    open(name + '.json', 'w').write(json_string)
    model.save_weights(name + '_weights.h5', overwrite=True)
    pickle.dump(history, open(name + '_history.p', 'wb'), protocol=4)


def create_model(input_shape, dropout = 0.6, lr=0.001, decay=1e-6):
    inp = Input(shape=input_shape)

    # x = GlobalAveragePooling2D()(inp)
    x = MaxPooling2D((4,4), strides=(4,4))(inp)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    x = Dense(512)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x1 = Dense(128)(x)
    x1 = PReLU()(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout)(x1)

    x1 = Dense(128)(x1)
    x1 = PReLU()(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout)(x1)

    x2 = Dense(512)(x)
    x2 = PReLU()(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(dropout)(x2)

    x2 = Dense(512)(x2)
    x2 = PReLU()(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(dropout)(x2)

    bbox_predictions1 = Dense(2, activation='linear', name='bbox_tl')(x1)
    bbox_predictions2 = Dense(2, activation='linear', name='bbox_br')(x1)
    fish_predictions = Dense(8, activation='softmax', name='class')(x2)

    '''
    eddigi legjobb: 640x480, maxpool, bn, flatten
    x1: 256,prelu,bn,do; 256,prelu,bn,do,bp1,bp2
    x2: 256,prelu,bn,do; 256,prelu,bn,do,fp
    do: 0.5
    lr=0.05, decay=1e-6
    epoch: 10

    other best: save_model(model, 6, 'val_class_acc_9078')
    epoch: 40
    dropout=0.7, lr=0.04, decay=1e-6
    '''

    model = Model(input=inp, output=[fish_predictions,
                                     bbox_predictions1,
                                     bbox_predictions2]) 

    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss=['categorical_crossentropy', 'mse', 'mse'],
                  loss_weights=[1.0, 0.00001, 0.00001],
                  metrics=['accuracy'])
    return model


def get_class_w(y):
    class_w = {}
    max_factor = max([len(np.where(y.argmax(1) == i)[0]) for i in range(len(classes))])

    for i in range(len(classes)):
        l = len(np.where(y.argmax(1) == i)[0])
        f = max_factor / l
        class_w[i] = f
        wf = f / class_w[i]
        percent = l / (len(y) / 100.)
        print('class: {}\t %: {:.2f}\t n: {} [{:.2f} -> {:.2f}]'.format(classes[i], percent, l, f, wf))
    return class_w


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', help='Preprocess training/testing data',
                        action='store_true')
    parser.add_argument('--info', help='Model info string', default='fish')
    parser.add_argument('--index', help='Model index', default=0)
    args = parser.parse_args()

    if args.preprocess:
        print('Preprocessing traingin/testing data...')
        boxes = read_bbox_annotations(config.bbox_annotations_path)
        relabels = read_relabels('relabels.csv')

        X_train, y_train, y_train_box = \
            read_training_images(config.training_images_path, boxes, relabels)

        X_test, Id_test = \
            read_testing_images(config.testing_images_path)

        print('Preprocessing X_train...')
        vgg = create_vgg(config.img_w, config.img_h)
        X_train_feat = predict(X_train, vgg, 8)

        print('Preprocessing X_test...')
        X_test_feat = predict(X_test, vgg, 8)

        print('Saving training data...')
        utils.save_array('X_train.bcolz', X_train)
        utils.save_array('y_train.bcolz', y_train)
        utils.save_array('y_train_box.bcolz', y_train_box)

        utils.save_array('X_train_feat.bcolz', X_train_feat)
        utils.save_array('X_test_feat.bcolz', X_test_feat)
        utils.save_array('X_test.bcolz', X_test)
        utils.save_array('Id_test.bcolz', Id_test)
    else:
        print('Reading preprocessed data...')
        X_train_feat = utils.load_array('X_train_feat.bcolz')
        y_train = utils.load_array('y_train.bcolz')
        y_train_box = utils.load_array('y_train_box.bcolz')
        X_test_feat = utils.load_array('X_test_feat.bcolz')
        Id_test = utils.load_array('Id_test.bcolz')
        print('Done')

        callbacks = [
            # CSVLogger('bbox_regression_' + time_str + '.csv', separator=',', append=False),
            # ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            #                 save_best_only=True,
            #                 monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience),
        ]

        ss = StratifiedShuffleSplit(n_splits=2,
                                    test_size=config.validation_split)
        train_idx, valid_idx = next(ss.split(X_train_feat, y_train.argmax(1)))

        # get class weights
        class_w = get_class_w(y_train[train_idx])

        # separate bbox
        y_box_tl = []
        y_box_wh = []
        for item in y_train_box:
            y_box_tl.append([item[0], item[1]])
            y_box_wh.append([item[2], item[3]])
        y_box_tl = np.array(y_box_tl)
        y_box_wh = np.array(y_box_wh)

        # create and train model
        print('Creating model...')
        model = create_model(np.shape(X_train_feat)[1:],
                             dropout=config.dropout,
                             lr=config.lr,
                             decay=config.decay)
        model.summary()

        print('Training...')
        history = model.fit(X_train_feat[train_idx],
                            [y_train[train_idx], y_box_tl[train_idx], y_box_wh[train_idx]],
                            batch_size=config.batch_size,
                            nb_epoch=config.nb_epoch,
                            validation_data=(X_train_feat[valid_idx],
                                             [y_train[valid_idx],
                                              y_box_tl[valid_idx],
                                              y_box_wh[valid_idx]]),
                            verbose=1, class_weight=class_w)

        print('Training finished')
        print('Saving model...')
        save_model(model, history.history, args.info, args.index)

        print('Testing...')

        y_test, y_test_bb_tl, y_test_bb_wh = run_test([model],
                                                      X_test_feat,
                                                      config.batch_size)
        save_test_result(y_test, Id_test, 'image', args.info, args.index)
        utils.save_array('y_test_bb_tl.bcolz', y_test_bb_tl)
        utils.save_array('y_test_bb_wh.bcolz', y_test_bb_wh)
