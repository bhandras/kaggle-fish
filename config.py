
training_images_path = 'train'
testing_images_path = 'test_stg1'
bbox_annotations_path = 'weijie_kaggle/NCFM/datasets'

img_w = 448
img_h = 448
max_boxes = 4
random_state = 42

# model parameters
validation_split=0.3
batch_size = 32
nb_epoch = 10
early_stopping_patience = 5
dropout = 0.7
lr = 0.04
decay = 1e-6
