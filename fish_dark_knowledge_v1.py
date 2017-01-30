# coding: utf-8
__author__ = 'bhandras: https://kaggle.com/bhandras'

import numpy as np
import os
import glob
# import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

from keras.models import Model, Sequential
from keras.layers import Input, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from keras.utils import np_utils
from keras import __version__ as keras_version

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, activity_l1

import pickle

# globals
np.random.seed(2017)
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
img_w = 224
img_h = 224
batch_size = 64
nb_epoch = 100
random_state = 42
num_folds = 3
dark_knowledge_file='submission_clip_folds_3_2017-01-25-15-40.csv'


def load_training_data(path):
    X_train = []
    y_train = []
    t0 = time.time()
    print('Reading training data...')

    soft_labels = pd.read_csv(dark_knowledge_file)
    values = soft_labels.ix[:, 'ALB':'YFT'].values
    image_names = soft_labels['image']
    ideal_prediction = {}
    for item in range(len(image_names)):
        ideal_prediction[image_names[item]] = values[item]
    for class_folder in classes:
        image_paths = glob.glob(os.path.join(path, class_folder, '*.jpg'))
        class_index = classes.index(class_folder)
        print('Loading class: {}'.format(class_folder))

        for image_path in image_paths:
            print(image_path)
            img = image.load_img(image_path, target_size=(img_w, img_h))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x=np.squeeze(x)
            X_train.append(x)
            if (image_path[-13:] in ideal_prediction):
                y_train.append(ideal_prediction[image_path[-13:]])
            else:
                tmp=np.zeros(8)
                tmp[4] = 1
                y_train.append(tmp)


    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.uint8)

    t1 = time.time()
    print('Reading finished: {} seconds'.format(round(t1 - t0, 2)))
    print('Training data shape:', X_train.shape)
    return X_train, y_train




def load_test_data(path):
    X_test = []
    Id_test = []
    t0 = time.time()

    image_paths = glob.glob(os.path.join(path, '*.jpg'))
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(img_w, img_h))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = np.squeeze(x)
        X_test.append(x)
        Id_test.append(os.path.basename(image_path))

    X_test = np.array(X_test)
    t1 = time.time()
    print('Test data shape:', X_test.shape)
    print('Reading finished: {} seconds'.format(round(t1 - t0, 2)))
    return X_test, Id_test


def create_submission(ids, predictions, info):
    result = pd.DataFrame(predictions, columns=classes)
    result.loc[:, 'image'] = pd.Series(ids, index=result.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result.to_csv(sub_file, index=False)


def create_model():
    input_tensor = Input(shape=(img_w, img_h, 3))
    model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    for layer in range(len(model.layers)-2):
        model.layers[layer].trainable = False


    x = model.output

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(8, activation='softmax', activity_regularizer=activity_l1(0.01))(x)
    
    model = Model(input=model.input, output=predictions)
    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def run_cross_validation_create_models(X_train, y_train, nfolds=10):
    datagen = image.ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    # datagen.mean = np.array([128, 128, 128], dtype=np.float32).reshape((3, 1, 1))

    skf = StratifiedKFold(y=y_train.argmax(1),
                          n_folds=nfolds,
                          shuffle=True,
                          random_state=random_state)
    curr_fold = 0
    sum_score = 0
    models = []

    for train_idx, valid_idx in skf:
        model = create_model()
        #model.load_weights('cache_v3/model_weights'+str(curr_fold)+'xception1.h5')
        print('Start StratifiedKFiold # {}/{}'.format(curr_fold, nfolds))
        print('Training split:', len(X_train[train_idx]))
        print('Validation split:', len(X_train[valid_idx]))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, verbose=1),
            ModelCheckpoint("tmp/model_advanced-{epoch:02d}-{val_loss:.4f}.h5", monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False,
                            mode='auto', period=1)
            # TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False),
        ]

        model.fit_generator(datagen.flow(X_train[train_idx],
                                         y_train[train_idx],
                                         batch_size=batch_size),
                            nb_epoch=nb_epoch,
                            samples_per_epoch=len(X_train),
                            verbose=1,
                            validation_data=datagen.flow(X_train[valid_idx],
                                                         y_train[valid_idx]),
                            nb_val_samples=len(valid_idx),
                            callbacks=callbacks)

        validation_predictions = model.predict(X_train[valid_idx],
                                               batch_size=batch_size,
                                               verbose=1)

        score = log_loss(y_train[valid_idx], validation_predictions)
        print('Score log_loss: ', score)
        sum_score += score * len(valid_idx)
        models.append(model)
        save_model(model, curr_fold, 'vgg16')

    score = sum_score / len(X_train)
    print("Log_loss train independent avg: ", score)
    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)

    return info_string, models


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def run_cross_validation_process_test(path, info_string, models):
    test_predictions = []
    X_test, y_test = load_test_data(path)

    for i in range(len(models)):
        print('Testing model # {}/{}'.format(i + 1, len(models)))
        model_prediction = models[i].predict(X_test,
                                             batch_size=batch_size,
                                             verbose=1)
        test_predictions.append(model_prediction)

    y_pred = merge_several_folds_mean(test_predictions, len(models))
    create_submission(y_test,
                      y_pred,
                      'loss_' + info_string + '_folds_' + str(len(models)))



if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    X_train, y_train = load_training_data('data/train')

    info_string, models = run_cross_validation_create_models(X_train,
                                                             y_train,
                                                             num_folds)

    run_cross_validation_process_test('data/test_stg1', info_string, models)

