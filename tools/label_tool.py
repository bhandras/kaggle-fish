import pandas as pd
import numpy as np
import os
import shutil

orig = 'test_stg1'
sep = 'test_stg1_sep'

if not os.path.isdir(sep):
    os.mkdir(sep)

df = pd.read_csv('submission_loss_split_model_folds_1_2017-02-01-17-57.csv')
print(df.columns)
for index, row in df.iterrows():
    img = row['image']
    c = np.argmax(row[:-1])
    print(img, c)
    class_dir = os.path.join(sep, c)
    if not os.path.isdir(class_dir):
        os.mkdir(class_dir)
    shutil.copyfile(os.path.join(orig, img), os.path.join(class_dir, img))
