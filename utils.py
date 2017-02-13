import os
import glob
import json
import time
import pickle
import datetime

import numpy as np
import bcolz

from sklearn.model_selection import StratifiedShuffleSplit


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def save_bbox_img(img, bbox_arr, filename):
    new_img = Image.new('RGB', (img.width, img.height), (255, 255, 255))
    new_img.paste(img)
    draw = ImageDraw.Draw(new_img)
    for i in range(0, len(bbox_arr), 4):
        bbox = (bbox_arr[i + 0], bbox_arr[i + 1], bbox_arr[i + 2], bbox_arr[i + 3])
        draw.line((bbox[0], bbox[1], bbox[2], bbox[1]), fill=(255, 0, 0))
        draw.line((bbox[2], bbox[1], bbox[2], bbox[3]), fill=(255, 0, 0))
        draw.line((bbox[2], bbox[3], bbox[0], bbox[3]), fill=(255, 0, 0))
        draw.line((bbox[0], bbox[3], bbox[0], bbox[1]), fill=(255, 0, 0))
    del draw
    new_img.save(open(filename, 'wb'))


def read_bbox_json(filename):
    boxes = {}
    with open(filename) as f:
        json_data = json.load(f)
        for item in json_data:
            fn = item['filename']
            cnt = 0
            for annotation in item['annotations']:
                x = annotation['x']
                y = annotation['y']
                w = annotation['width']
                h = annotation['height']
                label = annotation['class']
                if not fn in boxes:
                    boxes[fn] = {}
                if not label in boxes[fn]:
                    boxes[fn][label] = []
                boxes[fn][label].append([x, y, w, h])
    return boxes
