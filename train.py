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
import bcolz

from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam, Nadam, Adamax, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.layers import Input, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.regularizers import l1, activity_l1

from sklearn.metrics import log_loss
from keras.layers.advanced_activations import PReLU
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


def create_model_bb(input_shape, dropout = 0.6, lr=0.001, decay=1e-6):
    inp = Input(shape=input_shape)

    # x = GlobalAveragePooling2D()(inp)
    # x = MaxPooling2D((2,2))(inp)
    x = BatchNormalization()(inp)
    x = Flatten()(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(dropout)(x)

    x1 = Dense(512)(x)
    x1 = BatchNormalization()(x1)
    x1 = PReLU()(x1)
    x1 = Dropout(dropout)(x1)

    x1 = Dense(128)(x1)
    x1 = PReLU()(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout)(x1)

    x2 = Dense(512)(x)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    x2 = Dropout(dropout)(x2)

    x2 = Dense(256)(x2)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    x2 = Dropout(dropout)(x2)

    bbox_predictions1 = Dense(2, activation='linear', name='bbox_tl')(x1)
    bbox_predictions2 = Dense(2, activation='linear', name='bbox_wh')(x1)
    fish_predictions = Dense(8, activation='softmax', name='class',
                             activity_regularizer=activity_l1(0.01))(x2)

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
                  loss_weights=[1.0, 0.001, 0.001],
                  metrics=['accuracy'])
    return model


def create_model(input_shape, dropout = 0.6, lr=0.001, decay=1e-6):
    inp = Input(shape=input_shape)

    '''
    # fully convolutional
    x = BatchNormalization()(inp)
    x = Convolution2D(256, 3, 3, border_mode='same')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, border_mode='same')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, border_mode='same')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((1, 2))(x)

    x = Convolution2D(8, 3, 3, border_mode='same')(x)
    x = Dropout(dropout)(x)
    x = GlobalAveragePooling2D()(x)
    fish_predictions = Activation('softmax')(x)
    '''

    x = BatchNormalization()(inp)
    x = Flatten()(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(dropout)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(dropout)(x)
    
    fish_predictions = Dense(8, activation='softmax', name='class')(x)
    model = Model(input=inp, output=fish_predictions) 
    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])
    return model


def predict(X, model, batch_size):
    y_pred = []
    for batch in range(0, len(X), batch_size):
        y_pred.extend(model.predict(X[batch:batch + batch_size]))
    return np.array(y_pred)


def run_test(models, X, ids, batch_size, id_name, bb, info_string, index):
    df_labels = pd.DataFrame(columns=classes)
    df_bboxes = pd.DataFrame(columns=['x', 'y', 'w', 'h'])

    for i in range(len(models)):
        print('Testing model # {}/{}'.format(i+1, len(models)))
        y = models[i].predict(X, batch_size=batch_size, verbose=1)
        if bb:
            df_labels = pd.concat([df_labels,
                                   pd.DataFrame(y[0], columns=df_labels.columns)])
            df_bboxes = pd.concat([df_bboxes, pd.DataFrame(
                [a + b for a,b in zip(y[1], y[2])],
                columns=df_bboxes.columns)])
        else:
            df_labels = pd.concat([df_labels, pd.DataFrame(y, columns=classes)])
    df_labels.loc[:, id_name] = pd.Series(np.tile(ids, len(models)))
    df_labels = df_labels.groupby(id_name).mean().reset_index()
    if bb:
        df_bboxes.loc[:, id_name] = pd.Series(np.tile(ids, len(models)))
        df_boxes = df_boxes.groupby(id_name).mean().reset_index()

    print(df_labels.head())
    # save df_labels
    now = datetime.datetime.now()
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    filename = os.path.join('cache', info_string + '_' + str(index) + '_' +
                            str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv')
    print('Saving test result to: ', filename)
    df_labels.to_csv(filename, index=False)

    return df_labels, df_bboxes


def save_model(model, history, info_string, index):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    name = os.path.join('cache', info_string + '_' + str(index))
    open(name + '.json', 'w').write(json_string)
    model.save_weights(name + '_weights.h5', overwrite=True)
    pickle.dump(history, open(name + '_history.p', 'wb'), protocol=4)



def get_class_w(y):
    class_w = {}
    max_factor = max([len(np.where(y.argmax(1) == i)[0]) for i in range(len(classes))])

    for i in range(len(classes)):
        n = len(np.where(y.argmax(1) == i)[0])
        class_w[i] = max_factor / n
        percent = n / (len(y) / 100.0)
        print('{}\t{:.2f}%\t\tn: {}\t\tw: {:.2f}'
              .format(classes[i], percent, n, class_w[i]))
    return class_w


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', help='Model info string', default='fish')
    parser.add_argument('--index', help='Model index', default=0)
    parser.add_argument('--bb', help='Train BB regressor', default=False)
    parser.add_argument('--test', help='Test only', default=False)
    args = parser.parse_args()

    print('Reading preprocessed data...')
    X_train_feat = bcolz.open('train_images_feat.bc')
    y_train = bcolz.open('train_labels.bc')
    y_train_box = bcolz.open('train_bboxes.bc')
    X_test_feat = bcolz.open('test_stg1_images_feat.bc')
    Id_test = bcolz.open('test_stg1_ids.bc')
    print('Done')

    def scheduler(epoch):
        if epoch >= 4:
            return 0.001
        return config.lr

    callbacks = [
        LearningRateScheduler(scheduler),
        EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience),
    ]

    ss = StratifiedShuffleSplit(n_splits=2,
                                test_size=config.validation_split)
    train_idx, valid_idx = next(ss.split(X_train_feat, np.argmax(y_train, 1)))
    print('split',len(train_idx), len(valid_idx))
    # get class weights
    class_w = get_class_w(y_train[train_idx])

    # separate bboxes
    y_box_tl = []
    y_box_wh = []
    for item in y_train_box:
        y_box_tl.append([item[0], item[1]])
        y_box_wh.append([item[2], item[3]])
    y_box_tl = np.array(y_box_tl)
    y_box_wh = np.array(y_box_wh)

    # create and train model
    print('Creating model...')
    if args.bb:
        model = create_model_bb(np.shape(X_train_feat)[1:],
                                dropout=config.dropout,
                                lr=config.lr,
                                decay=config.decay)
    else:
        model = create_model(np.shape(X_train_feat)[1:],
                             dropout=config.dropout,
                             lr=config.lr,
                             decay=config.decay)
        model.summary()

    if not args.test:
        print('Training...')
        if args.bb:
            history = model.fit(X_train_feat[train_idx],
                                [y_train[train_idx], y_box_tl[train_idx], y_box_wh[train_idx]],
                                batch_size=config.batch_size,
                                nb_epoch=config.nb_epoch,
                                validation_data=(X_train_feat[valid_idx],
                                                 [y_train[valid_idx],
                                                  y_box_tl[valid_idx],
                                                  y_box_wh[valid_idx]]),
                                verbose=1,
                                class_weight=class_w,
                                callbacks=callbacks)
        else:
            history = model.fit(X_train_feat[train_idx],
                                y_train[train_idx],
                                batch_size=config.batch_size,
                                nb_epoch=config.nb_epoch,
                                validation_data=(X_train_feat[valid_idx],
                                                 y_train[valid_idx]),
                                verbose=1,
                                class_weight=class_w,
                                callbacks=callbacks)


        print('Training finished')
        print('Saving model...')
        save_model(model, history.history, args.info, args.index)
    else:
        model.load_weights('cache/fish_0_weights.h5')

    print('Testing...')
    df_y_test, df_y_test_bbox = run_test([model],
                                         X_test_feat,
                                         Id_test,
                                         config.batch_size,
                                         'image',
                                         args.bb,
                                         args.info,
                                         args.index)
    utils.save_array('y_test_bbox.bc', np.array(df_y_test_bbox))
