import numpy as np
import bcolz

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input

import utils
import config

def create_vgg(w, h):
    model = VGG16(include_top=False, weights='imagenet',
                  input_tensor=Input(shape=(h, w, 3)))
    for layer in model.layers:
        layer.trainable = False
    return model


def predict_vgg(X, model, batch_size):
    total = len(X)
    percent = 0.0
    last_percent = 0
    y = []
    for i in range(0, len(X), batch_size):
        y_batch = preprocess_input(np.asarray(X[i:i + batch_size], dtype=np.float32))
        y.extend(model.predict(y_batch))
        percent += len(y_batch) / (total / 100.0)
        if int(percent) > last_percent:
            last_percent = int(percent)
            print(last_percent,'%')
    return np.array(y)


vgg = create_vgg(config.img_w, config.img_h)
batch_size = 64
print('Preprocessing train_images...')
train_images = bcolz.open('train_images.bc')
train_images_feat = predict_vgg(train_images, vgg, batch_size)
utils.save_array('train_images_feat.bc', train_images_feat)
print('Done')
print('Preprocessing test_images...')
test_images = bcolz.open('test_stg1_images.bc')
test_images_feat = predict_vgg(test_images, vgg, batch_size)
utils.save_array('test_stg1_images_feat.bc', test_images_feat)
print('Done')

