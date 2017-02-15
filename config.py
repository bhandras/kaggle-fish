
training_images_path = 'train'
testing_images_path = 'test_stg1'
bbox_annotations_path = 'weijie_kaggle/NCFM/datasets'

img_w = 448
img_h = 448
random_state = 42

augmentation_factor = 10
augmentation_x_scale_range = 1
augmentation_y_scale_range = 1
augmentation_rotation_range = 0.4
augmentation_translation_range = 100

# model parameters
validation_split=0.3
batch_size = 32
nb_epoch = 50
early_stopping_patience = 7
dropout = 0.6
lr = 0.001
decay = 1e-6
