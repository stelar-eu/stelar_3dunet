
import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
import tifffile


#Make sure the GPU is available. 
import tensorflow as tf

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

crop_types_all_list = [ 1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 41]


export CUDA_VISIBLE_DEVICES=1
conda deactivate
conda deactivate
cd stelar_3dunet/
source spt19/bin/activate
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 4 --crop_2 5 --crop_3 7
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 8 --crop_2 9 --crop_3 10
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 11 --crop_2 12 --crop_3 13
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 14 --crop_2 15 --crop_3 16
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 18 --crop_2 19 --crop_3 20
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 21 --crop_2 23 --crop_3 27
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 28 --crop_2 30 --crop_3 32
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 33 --crop_2 34 --crop_3 35
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 36 --crop_2 37 --crop_3 40
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 37 --crop_2 40 --crop_3 41


chosen_crop_types_list_list = [[1, 2, 3], [4, 5, 7], [8,  9, 10], [11, 12, 13], [14, 15, 16], [18, 19, 20], [21, 23, 27], [28, 30, 32], [33, 34, 35], [36, 37, 40], [37, 40, 41]]

export CUDA_VISIBLE_DEVICES=0
conda deactivate
conda deactivate
cd stelar_3dunet/
conda activate stcon4
python3 test_set_saving_10000_spt_points_simpler.py --crop_1 34 --crop_2 36 --crop_3 37

python3 test_set_saving_10000_spt_points_simpler.py --crop_1 33 --crop_2 36 --crop_3 40


    chosen_crop_types_list_list = [[34, 36, 37], [33, 36, 40], [33, 36, 41]]

'''



import segmentation_models_3D as sm


from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



'''physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)'''



num_epochs = 450
vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}

import argparse

parser = argparse.ArgumentParser(description='Enter crop type numbers in order')
parser.add_argument('--crop_1', type=int, default=1, help='Select crop type')
parser.add_argument('--crop_2', type=int, default=2, help='Select crop type')
parser.add_argument('--crop_3', type=int, default=3, help='Select crop type')


args = parser.parse_args()

cr_1 = args.crop_1
cr_2 = args.crop_2
cr_3 = args.crop_3


chosen_crop_types_list = [cr_1, cr_2, cr_3]

crop_types_all_list = [ 1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 41]


print("observe")

print("vista_crop_dict[cr_1]", vista_crop_dict[cr_1])
print("vista_crop_dict[cr_2]", vista_crop_dict[cr_2])
print("vista_crop_dict[cr_3]", vista_crop_dict[cr_3])

print("vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[1]]", vista_crop_dict[chosen_crop_types_list[0]], vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[2]])

sampling_group_fractions = [1.0, 1.0, 1.0]


#all_input_img = []
#all_input_mask = []

all_input_img_f = []
all_input_mask_f = []
counted = 0
for crop_no in chosen_crop_types_list:
    chosen_crop_type = vista_crop_dict[crop_no]
    print("chosen_crop_type", chosen_crop_type)
    '''input_img = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train_'+vista_crop_dict[crop_no]+'.tif')
    input_mask = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab_'+vista_crop_dict[crop_no]+'.tif').astype(np.uint8)

    print("input_img.shape", input_img.shape)
    print("input_mask.shape", input_mask.shape)'''

    input_img0 = io.imread('./storage/per_crop_data_labels_test/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n.tif')#[:9000]
    input_mask0 = io.imread('./storage/per_crop_data_labels_test/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n.tif').astype(np.uint8)#[:9000]
    print("input_img0.shape", input_img0.shape)
    print("input_mask0.shape", input_mask0.shape)


    '''input_img1 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n1.tif')
    input_mask1 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n1.tif').astype(np.uint8)
    print("input_img1.shape", input_img1.shape)
    print("input_mask1.shape", input_mask1.shape)

    input_img2 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n2.tif')
    input_mask2 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n2.tif').astype(np.uint8)
    print("input_img2.shape", input_img2.shape)
    print("input_mask2.shape", input_mask2.shape)'''

    '''input_img3 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n3.tif')
    input_mask3 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n3.tif').astype(np.uint8)
    print("input_img3.shape", input_img3.shape)
    print("input_mask3.shape", input_mask3.shape)

    input_img4 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n4.tif')
    input_mask4 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n4.tif').astype(np.uint8)
    print("input_img4.shape", input_img4.shape)
    print("input_mask4.shape", input_mask4.shape)

    input_img5 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n5.tif')
    input_mask5 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n5.tif').astype(np.uint8)
    print("input_img5.shape", input_img5.shape)
    print("input_mask5.shape", input_mask5.shape)

    input_img6 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n6.tif')
    input_mask6 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n6.tif').astype(np.uint8)
    print("input_img6.shape", input_img6.shape)
    print("input_mask6.shape", input_mask6.shape)

    input_img7 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n7.tif')
    input_mask7 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n7.tif').astype(np.uint8)
    print("input_img7.shape", input_img7.shape)
    print("input_mask7.shape", input_mask7.shape)

    input_img8 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n8.tif')
    input_mask8 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n8.tif').astype(np.uint8)
    print("input_img8.shape", input_img8.shape)
    print("input_mask8.shape", input_mask8.shape)

    input_img9 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n9.tif')
    input_mask9 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n9.tif').astype(np.uint8)
    print("input_img9.shape", input_img9.shape)
    print("input_mask9.shape", input_mask9.shape)

    input_img10 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n10.tif')
    input_mask10 = io.imread('./storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n10.tif').astype(np.uint8)
    print("input_img9.shape", input_img9.shape)
    print("input_mask9.shape", input_mask9.shape)'''


    '''input_img_f = np.concatenate((input_img, input_img0, input_img1, input_img2, input_img3, input_img4, input_img5, input_img6, input_img7, input_img8, input_img9, input_img10), axis=0)
    input_mask_f = np.concatenate((input_mask, input_mask0, input_mask1, input_mask2, input_mask3, input_mask4, input_mask5, input_mask6, input_mask7, input_mask8, input_mask9, input_mask10), axis=0)'''

    """input_img_f = np.concatenate(( input_img0), axis=0)
    input_mask_f = np.concatenate(( input_mask0), axis=0)"""

    input_img_f =input_img0
    input_mask_f = input_mask0

    '''input_img_f = np.concatenate((input_img, input_img0, input_img1, input_img2, input_img3, input_img4, input_img5, input_img6, input_img7, input_img8, input_img9), axis=0)
    input_mask_f = np.concatenate((input_mask, input_mask0, input_mask1, input_mask2, input_mask3, input_mask4, input_mask5, input_mask6, input_mask7, input_mask8, input_mask9), axis=0)'''

    bis = int(len(input_img_f)*sampling_group_fractions[counted]) - 2

    print("bis", bis)
    print("before input_img_f.shape", input_img_f.shape)
    print("before input_mask_f.shape", input_mask_f.shape)

    input_img_f = input_img_f[:bis]
    input_mask_f = input_mask_f[:bis]

    print("after input_img_f.shape", input_img_f.shape)
    print("after input_mask_f.shape", input_mask_f.shape)

    all_input_img_f.append(input_img_f)
    all_input_mask_f.append(input_mask_f)
    counted+=1    
all_input_img_f = np.concatenate((all_input_img_f), axis=0)
all_input_mask_f = np.concatenate((all_input_mask_f), axis=0)


print("all_input_img_f.shape", all_input_img_f.shape)
print("all_input_mask_f.shape", all_input_mask_f.shape)

'''input_img = np.concatenate((all_input_img_f), axis=0).reshape(-1, 64, 64, 64)
input_mask = np.concatenate((all_input_mask_f), axis=0).reshape(-1, 64, 64)'''

input_img = all_input_img_f
input_mask = all_input_mask_f


print("1 input_img.max()", input_img.max())
print("1 input_img.min()", input_img.min())

#input_img = ((input_img*1000)//10)/100

#input_img = input_img/10

#input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

print("2 input_img.max()", input_img.max())
print("2 input_img.min()", input_img.min())


del all_input_img_f
del all_input_mask_f
del input_img_f
del input_mask_f

#unique_elements, element_counts = np.unique(input_mask, return_counts=True)

#print("1 : unique_elements, element_counts", unique_elements, element_counts)

#selected_counts = np.array([element_counts[1], element_counts[2], element_counts[3]])




input_mask = np.repeat(input_mask[:, np.newaxis, :, :], repeats=64, axis=1)

#np.random.shuffle(input_mask)
#np.random.shuffle(input_img)

#input_img = input_img[:10000]
#input_mask = input_mask[:10000]

print("final input_img.shape", input_img.shape)
print("final input_mask.shape", input_mask.shape)


#all_input_img = 0
#all_input_mask = 0

# this should not be done whwen you want ground truth for inference. This is only for training.
print("1 : input_mask.shape", input_mask.shape) 
'''input_mask_all = input_mask.copy()
input_mask[input_mask<chosen_crop_types_list[0]]=0
input_mask[input_mask>chosen_crop_types_list[-1]]=0
for i in range(3):
    input_mask[input_mask==chosen_crop_types_list[i]]=i+1'''

print("2 : input_mask.shape", input_mask.shape) 


'''lai_uniques = np.unique(input_img)

for n in range(len(lai_uniques)):
  input_img[input_img==lai_uniques[n]]=n
input_img = input_img.astype(np.uint8)'''


lai_uniques = 0
n_classes=4


train_img = np.stack((input_img,)*3, axis=-1)
'''train_mask = np.expand_dims(input_mask, axis=4)
print("3 : train_mask.shape", train_mask.shape) 

train_mask_cat = to_categorical(train_mask, num_classes=n_classes)
print("4 : train_mask_cat.shape", train_mask_cat.shape)'''

#X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.10, random_state = 0)

#del X_train
#del y_train

X_train, X_test, y_train, y_test_all = train_test_split(train_img, input_mask, test_size = 0.10, random_state = 0)

del X_train
del y_train

#X_train = X_train[:9000]
#y_train = y_train[:9000]
X_test = X_test[:1000]
#y_test = y_test[:1000]
y_test_all = y_test_all[:1000]


#print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)
#print("y_train.shape", y_train.shape)
print("y_test_all.shape", y_test_all.shape)

# now save the test set
tifffile.imsave('./storage/test_sets_of_subsets_all/X_test'+'_contains'+vista_crop_dict[cr_1]+'_'+vista_crop_dict[cr_2]+'_'+vista_crop_dict[cr_3]+'_.tif', X_test)
#tifffile.imsave('./storage/test_sets_of_subsets_all/y_test'+'_contains'+vista_crop_dict[cr_1]+'_'+vista_crop_dict[cr_2]+'_'+vista_crop_dict[cr_3]+'_.tif', y_test)
tifffile.imsave('./storage/test_sets_of_subsets_all/y_test_all'+'_contains'+vista_crop_dict[cr_1]+'_'+vista_crop_dict[cr_2]+'_'+vista_crop_dict[cr_3]+'_.tif', y_test_all)