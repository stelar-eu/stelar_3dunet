
import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)


#Make sure the GPU is available. 
#import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


'''
conda remove -n spt16 --all

conda deactivate
conda deactivate
cd stelar_3d/
virtualenv spt20
source spt19/bin/activate


pip install tensorflow
pip install classification-models-3D
pip install efficientnet-3D
pip install segmentation-models-3D
pip install scikit-learn
pip install matplotlib
pip install patchify
pip install scikit-image

python 3D_unet_1.py 



pip uninstall tensorflow
pip uninstall classification-models-3D
pip uninstall efficientnet-3D
pip uninstall segmentation-models-3D
pip uninstall scikit-learn
pip uninstall matplotlib
pip uninstall patchify
pip uninstall scikit-image

'''


'''

export CUDA_VISIBLE_DEVICES=0
conda deactivate
conda deactivate
cd stelar_3d/
source spt19/bin/activate
python3 3D_unet_bias_mit_1_data_size_examine.py 


'''



import segmentation_models_3D as sm


from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


num_epochs = 1500
vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}

chosen_crop_types_list = [1, 2, 3]

print("vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[1]]", vista_crop_dict[chosen_crop_types_list[0]], vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[2]])

all_input_img = []
all_input_mask = []

chosen_crop_types_list = [ 1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 41]
for i in chosen_crop_types_list:
  input_img = io.imread('/home/luser/stelar_3d/storage/per_crop_data_labels/'+vista_crop_dict[i]+'/train_'+vista_crop_dict[i]+'.tif')
  input_mask = io.imread('/home/luser/stelar_3d/storage/per_crop_data_labels/'+vista_crop_dict[i]+'/lab_'+vista_crop_dict[i]+'.tif').astype(np.uint8)
  print("crop number", i)
  print("vista_crop_dict[i]", vista_crop_dict[i])
  print("input_img.shape", input_img.shape)
  print("input_mask.shape", input_mask.shape)
  print()



'''for i in chosen_crop_types_list:
    input_img = io.imread('/home/luser/stelar_3d/storage/per_crop_data_labels/'+vista_crop_dict[i]+'1/train'+vista_crop_dict[i]+'.tif')
    input_mask = io.imread('/home/luser/stelar_3d/storage/per_crop_data_labels/'+vista_crop_dict[i]+'1/lab'+vista_crop_dict[i]+'.tif').astype(np.uint8)

    print("vista_crop_dict[i]", vista_crop_dict[i])
    print("input_img.shape", input_img.shape)
    print("input_mask.shape", input_mask.shape)
    print()
'''