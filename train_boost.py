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
from keras.utils.generic_utils import Progbar

from sklearn.metrics import log_loss
from keras.layers.advanced_activations import PReLU
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import log_loss

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


def create_model(input_shape, dropout = 0.6, lr=0.001, decay=1e-6):
    inp = Input(shape=input_shape) 
    x = BatchNormalization()(inp)

    nf = 256

    # fully convolutional 
    x = Convolution2D(nf, 3, 3, border_mode='same', activation='relu')(x)
    # x = PReLU()(x)
    # x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Convolution2D(nf, 3, 3, border_mode='same', activation='relu')(x)
    # x = PReLU()(x)
    # x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Convolution2D(nf, 3, 3, border_mode='same', activation='relu')(x)
    # x = PReLU()(x)
    # x = MaxPooling2D((1, 2))(x)
    x = BatchNormalization()(x)

    x = Convolution2D(8, 3, 3, border_mode='same')(x)
    x = Dropout(dropout)(x)

    x = GlobalAveragePooling2D()(x)
    fish_predictions = Activation('softmax')(x)

    '''
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    # x = PReLU()(x)
    x = Dropout(dropout)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    # x = PReLU()(x)
    x = Dropout(dropout)(x)
    fish_predictions = Dense(8, activation='softmax', name='class')(x)
    '''
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
        # y = models[i].predict_proba(X, batch_size=batch_size)

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
    parser.add_argument('--test', help='Test only', default=False)
    args = parser.parse_args()

    print('Reading preprocessed data...')
    X_train_feat = bcolz.open('train_images_feat.bc')
    y_train = bcolz.open('train_labels.bc')
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

    # create and train model
    print('Creating model...')
    model = create_model(np.shape(X_train_feat)[1:],
                         dropout=config.dropout,
                         lr=config.lr,
                         decay=config.decay)
    model.summary()

    def adaboostweight(weight, errors, corrects):
        Z = 0
        for i in range(len(weight)):
            eps = 0.000001 + weight[i] * errors[i] / (errors[i] + corrects[i])
            alfa = 0.5 * np.log((1 - eps) / eps)
            weight[i] *= np.exp(alfa * np.sign(errors[i] - corrects[i]))
        Z = np.sum(weight)
        return weight / Z

    def BcolzRandomSplit(X, y, batch_size, split_at=0.8):
        class BcolzIter(object):
            def __init__(self, X, y, idx_array, batch_size):
                self.X = X
                self.y = y
                self.idx_array = idx_array
                self.batch_size = batch_size
                self.curr_chunk = 0
                self.batch_chunks = batch_size // X.chunklen

            def reset(self):
                self.curr_chunk = 0

            def len(self):
                return int(np.ceil(len(self.idx_array)) / self.batch_chunks)

            def next(self):
                if self.curr_chunk >= len(self.idx_array):
                    raise StopIteration()

                if self.curr_chunk == 0:
                    self.idx_array = np.random.permutation(self.idx_array)

                X_batch = []
                y_batch = []
                for k in range(self.batch_chunks):
                    if self.curr_chunk == X.nchunks:
                        X_batch.append(self.X.leftover_array[:self.X.leftover_elements])
                        curr_batch_size = self.X.leftover_elements
                    else:
                        X_batch.append(self.X.chunks[self.curr_chunk][:])
                        curr_batch_size = self.X.chunklen
                    y_start = self.curr_chunk * X.chunklen
                    y_end = y_start + curr_batch_size
                    y_batch.append(self.y[y_start:y_end])

                    self.curr_chunk += 1
                    if self.curr_chunk >= len(self.idx_array):
                        break

                X_batch = np.concatenate(X_batch)
                y_batch = np.concatenate(y_batch)
                return X_batch, y_batch

            def __iter__(self):
                return self

            def __next__(self, *args, **kwargs):
                return self.next(*args, **kwargs)

        # impl
        idx_array = np.random.permutation(X.nchunks + 1)
        split_idx = int(len(idx_array) * split_at)
        s1 = idx_array[0:split_idx]
        s2 = idx_array[split_idx:]
        return BcolzIter(X, y, s1, batch_size), BcolzIter(X, y, s2, batch_size)


    if not args.test:
        print('Training...')
        class_w = np.array(len(classes) * [1./len(classes)])
        train, valid = BcolzRandomSplit(X_train_feat,
                                        y_train,
                                        config.batch_size,
                                        (1.0 - config.validation_split))

        for epoch in range(config.nb_epoch):
            print('Epoch {} of {}'.format(epoch + 1, config.nb_epoch))
            train.reset()
            valid.reset()
            pbar = Progbar(train.len())

            for X_batch, y_batch in train:
                metrics = model.train_on_batch(X_batch, y_batch, class_weight=class_w)
                pbar.add(1, values=[m for m in zip(model.metrics_names,
                                                   metrics)])

            print('Validation...')
            pred = []
            real = []
            pbar = Progbar(valid.len())
            for X_batch, y_batch in valid:
                pred.extend(model.predict_on_batch(X_batch))
                real.extend(y_batch)
                pbar.add(1)

            print('pred', np.shape(np.asarray(pred)))
            print('real', np.shape(np.asarray(real)))
            print(real[0:5])
            fp = np.zeros(len(classes))
            fn = np.zeros(len(classes))
            tp = np.zeros(len(classes))
            tn = np.zeros(len(classes))

            pred_l = np.argmax(pred, axis=1)
            true_l = np.argmax(real, axis=1)
            for i in range(len(pred_l)):
                if pred_l[i] != true_l[i]:
                    fp[pred_l[i]] += 1
                    fn[true_l[i]] += 1
                else:
                    tp[pred_l[i]] += 1
            print('')
            print('tp', [item for item in zip(classes, tp)])
            print('fn', [item for item in zip(classes, fn)])
            print('fp', [item for item in zip(classes, fp)])

            recall = np.zeros(len(classes))
            accuracy = np.zeros(len(classes))
            precision = np.zeros(len(classes))

            for i in range(len(classes)):
                tn[i] = len(pred) - (tp[i] + fn[i] + fp[i])
                recall[i] = tp[i] / (tp[i] + fn[i])
                accuracy[i] = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])
                precision[i] = tp[i] / (tp[i] + fp[i])

            print('')
            print('rec:', [item for item in zip(classes, recall)])
            print('pre:', [item for item in zip(classes, precision)])
            print('acc:', [item for item in zip(classes, accuracy)])
            print(np.shape(np.array(real)))
            print(np.shape(np.array(pred)))
            print('val_loss:', log_loss(np.array(real), np.array(pred)))
            class_w = adaboostweight(class_w, fn, tp)
            print('class_w:')
            for item in zip(classes, class_w):
                print('{}: {}'.format(item[0], item[1]))

        print('Training finished')
        print('Saving model...')
        save_model(model, history.history, args.info, args.index)
    else:
        # model.load_weights('cache/boost_12_weights.h5')
        print('load test model here...')

    print('Testing...')
    df_y_test, _ = run_test([model],
                            X_test_feat,
                            Id_test,
                            config.batch_size,
                            'image',
                            False,
                            args.info,
                            args.index)
