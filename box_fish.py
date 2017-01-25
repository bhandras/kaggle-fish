import os
import glob
import json
import time
import pickle
import datetime

import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.model_selection import KFold

# NoF intentionally left out
training_data_path = 'train'
annotations_path = 'weijie_kaggle/NCFM/datasets'
classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
# classes = ['SHARK']
img_w = 299
img_h = 299
max_boxes = 4
batch_size = 64
nb_epoch = 100
early_stopping_patience = 10

def process_box_json(data, boxes):
    for item in data:
        filename = os.path.basename(item['filename'])
        cnt = 0
        for annotation in item['annotations']:
            x = annotation['x']
            y = annotation['y']
            w = annotation['width']
            h = annotation['height']
            c = annotation['class']
            if not c in boxes:
                boxes[c] = {}
            if not filename in boxes[c]:
                boxes[c][filename] = []
            boxes[c][filename].extend([x, y, x+w, y+h])


def read_boxes():
    boxes = {}
    for c in classes:
        with open(os.path.join(annotations_path, c + '.json')) as f:
            data = json.load(f)
            process_box_json(data, boxes)
    return boxes


def read_training_data(path):
    # {'ALB': {'image_path.jpg': [(x1, y1, x2, y2), ...], }}
    boxes = read_boxes()
    X_train = []
    y_train = []

    t0 = time.time()
    print('Reading training data...')

    for class_folder in classes:
        image_paths = glob.glob(os.path.join(path, class_folder, '*.jpg'))
        class_index = classes.index(class_folder)
        print('Loading class: {}'.format(class_folder))

        for image_path in image_paths:
            print('Reading: ', image_path)
            img = image.load_img(image_path)
            x_scale = float(img_w) / float(img.width)
            y_scale = float(img_h) / float(img.height)
            print('Scale factors (x, y): ', x_scale, y_scale)

            img = img.resize((img_h, img_w))
            img = image.img_to_array(img)
            X_train.append(img)

            has_boxes = False
            image_name = os.path.basename(image_path)
            if class_folder in boxes:
                if image_name in boxes[class_folder]:
                    image_boxes = boxes[class_folder][image_name][:max_boxes * 4]
                    diff = max_boxes * 4 - len(image_boxes)
                    image_boxes.extend(diff * [0])

                    # scale bounding boxes
                    for i in range(0, 4 * max_boxes, 2):
                        image_boxes[i] = x_scale * image_boxes[i]
                        image_boxes[i + 1] = y_scale * image_boxes[i + 1]

                    y_train.append(image_boxes)
                    has_boxes = True
            if not has_boxes:
                y_train.append(max_boxes * [0, 0, 0, 0])
            print(y_train[-1])

    X_train = np.array(X_train)
    preprocess_input(X_train)
    y_train = np.array(y_train, dtype=np.float32)

    t1 = time.time()
    print('Reading finished: {} seconds'.format(round(t1 - t0, 2)))
    print('Training data shape:', X_train.shape)
    return X_train, y_train


def create_model():
    input_tensor = Input(shape=(img_w, img_h, 3))
    model = VGG16(include_top=False, weights='imagenet',
                  input_tensor=input_tensor)

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(16, activation='linear')(x)

    model = Model(input=model.input, output=predictions)
    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
    return model


def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


if __name__ == '__main__':
    if os.path.isfile('train.p'):
        print('Loading train.p')
        X_train, y_train = pickle.load(open('train.p', 'rb'))
        print('Finished loading train.p')
    else:
        X_train, y_train = read_training_data(training_data_path)
        print('Saving train.p')
        pickle.dump((X_train, y_train), open('train.p', 'wb'), protocol=4)

    time_str = str(datetime.datetime.now()).replace(' ', '_')
    model = create_model()

    callbacks = [
            CSVLogger('bbox_regression_' + time_str + '.csv', separator=',',
                      append=False),
            ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}.hdf5',
                            save_best_only=True,
                            monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1),
    ]

    kf = KFold(n_splits=2)
    for train_idx, test_idx in kf.split(X_train):
        model.fit(X_train[train_idx],
                  y_train[train_idx],
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_train[test_idx], y_train[test_idx]),
                  callbacks=callbacks,
                  verbose=1)
    print('Exiting.. (https://github.com/fchollet/keras/issues/2110)')
    time.sleep(0.5)
