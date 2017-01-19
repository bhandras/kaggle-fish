import os
import glob
import json
from PIL import Image

# NoF intentionally left out
data_path = '../train'
annotations_path = '../weijie_kaggle/NCFM/datasets'
classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']

def process_annotations(data):
    for item in data:
        filename = item['filename']
        cnt = 0
        for annotation in item['annotations']:
            x = annotation['x']
            y = annotation['y']
            w = annotation['width']
            h = annotation['height']
            c = annotation['class']
            print(filename, x, y, x+w, y+h)
            img = Image.open(os.path.join(data_path, filename))
            base_filename = os.path.basename(filename)
            new_filename = filename[:-len(base_filename)] + 'clipped_' + \
                           str(cnt) + '_' + base_filename

            img.crop((x, y, x+w, y+h)).save(os.path.join('clipped',
                                                         new_filename))
            print(new_filename, 'done')
            cnt += 1
            # print(c, x, y, w, h)


for c in classes:
    class_path = os.path.join('clipped', c)
    if not os.path.isdir(class_path):
        os.mkdir(class_path)
    with open(os.path.join(annotations_path, c + '.json')) as f:
        data = json.load(f)
        process_annotations(data)
