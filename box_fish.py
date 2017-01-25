'''
    model ideas from:
        https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/27048/single-vgg16-pretrained-for-1-logloss
'''
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
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split

from PIL import Image, ImageDraw

# NoF intentionally left out
training_data_path = 'train'
annotations_path = 'weijie_kaggle/NCFM/datasets'
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
# classes = ['SHARK']
img_w = 224
img_h = 224
max_boxes = 4
batch_size = 64
nb_epoch = 100
early_stopping_patience = 5
debug = False


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
        json_file = os.path.join(annotations_path, c + '.json') 
        if not os.path.isfile(json_file):
            continue
        with open(json_file) as f:
            data = json.load(f)
            process_box_json(data, boxes)
    return boxes


def save_img(orig_img, bboxes, fname):
    img = Image.new('RGB', (orig_img.width, orig_img.height), (255, 255, 255))
    img.paste(orig_img)
    draw = ImageDraw.Draw(img)
    for i in range(0, len(bboxes), 4):
        bbox = (bboxes[i + 0], bboxes[i + 1], bboxes[i + 2], bboxes[i + 3])
        draw.line((bbox[0], bbox[1], bbox[2], bbox[1]), fill=(255, 0, 0))
        draw.line((bbox[2], bbox[1], bbox[2], bbox[3]), fill=(255, 0, 0))
        draw.line((bbox[2], bbox[3], bbox[0], bbox[3]), fill=(255, 0, 0))
        draw.line((bbox[0], bbox[3], bbox[0], bbox[1]), fill=(255, 0, 0))
    del draw
    img.save(open(fname, 'wb'))


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
            image_name = os.path.basename(image_path)

            img = image.load_img(image_path)
            x_scale = float(img_w) / float(img.width)
            y_scale = float(img_h) / float(img.height)
            print('Scale factors (x, y): ', x_scale, y_scale)
            img = img.resize((img_h, img_w))

            has_boxes = False
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

                    # save resized image with boxes
                    if debug:
                        save_img(img, image_boxes,
                                 os.path.join(os.path.dirname(image_path), 'alma_' +
                                              image_name))
            if not has_boxes:
                y_train.append(max_boxes * [0, 0, 0, 0])

            img = image.img_to_array(img)
            X_train.append(img)
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
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.2)(x)
    predictions = Dense(16, activation='linear')(x)

    model = Model(input=model.input, output=predictions)
    # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    # model.summary()
    return model


def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def load_and_test_model(model_path, test_path):
    model = create_model()
    model.load_weights(model_path)
    run_test(model, test_path)


def run_test(model, test_path):
    image_paths = glob.glob(os.path.join(test_path, '*.jpg'))
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(img_w, img_h))
        x = image.img_to_array(img)
        np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        model_prediction = model.predict(x)
        print(model_prediction)
        save_img(img, model_prediction,
                 os.path.join(os.path.dirname(image_path),
                              'predicted_' + os.path.basename(image_path)))


if __name__ == '__main__':
    if os.path.isfile('train.p'):
        print('Loading train.p')
        X, y = pickle.load(open('train.p', 'rb'))
        print('Finished loading train.p')
    else:
        X, y = read_training_data(training_data_path)
        print('Saving train.p')
        pickle.dump((X, y), open('train.p', 'wb'), protocol=4)

    time_str = str(datetime.datetime.now()).replace(' ', '_')
    model = create_model()

    callbacks = [
            # CSVLogger('bbox_regression_' + time_str + '.csv', separator=',', append=False),
            ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                            save_best_only=True,
                            monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience),
    ]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks,
              verbose=1)
    run_test(model, 'test_stg1')
    print('Exiting.. (https://github.com/fchollet/keras/issues/2110)')
    time.sleep(0.5)
