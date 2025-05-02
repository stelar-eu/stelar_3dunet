
import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
import gc


#Make sure the GPU is available. 
from tensorflow.keras.utils import Sequence
import tifffile

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

device_name = tf.test.gpu_device_name()


# very important  : export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/luser/miniforge3/envs/stcon3/lib/python3.11/site-packages/tensorflow/include/third_party/gpus/cuda/


'''if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))'''


'''
conda remove -n spt16 --all

conda deactivate
conda deactivate
cd stelar_3d/
virtualenv spt20
source spt19/bin/activate


python3 -m venv atrial

pip3 install tensorflow
pip3 install classification-models-3D
pip3 install efficientnet-3D
pip3 install segmentation-models-3D
pip3 install scikit-learn
pip3 install matplotlib
pip3 install patchify
pip3 install scikit-image

python ./extras/3D_unet.py


#######################################
pip3 install classification-models-3D==1.0.10
pip3 install efficientnet-3D==1.0.2
pip3 install segmentation-models-3D==1.0.7
pip3 install scikit-learn==1.5.0
pip3 install matplotlib==3.9.0
pip3 install patchify==0.2.3
pip3 install scikit-image==0.24.0
#######################################



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
cd stelar_3dunet/
source spt19/bin/activate
python3 3D_unet_data_generator_check_pt.py --crop_1 1 --crop_2 2 --crop_3 3


python3 3D_unet_data_generator_check_pt.py --crop_1 37 --crop_2 40 --crop_3 41




'''


# The new commands are here 

'''
export CUDA_VISIBLE_DEVICES=0
conda deactivate
conda deactivate
conda activate stcon3
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt.py --crop_1 1 --crop_2 2 --crop_3 3
python3 3D_unet_data_generator_check_pt.py --crop_1 4 --crop_2 5 --crop_3 7


'''

# advanced commands
'''

export CUDA_VISIBLE_DEVICES=0
conda deactivate
conda deactivate
conda activate /home/luser/miniforge3/envs/stcon3
cd stelar_3dunet/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/luser/miniforge3/envs/stcon3/lib/python3.11/site-packages/tensorflow/include/third_party/gpus/cuda/
python3 3D_unet_data_generator_check_pt.py --crop_1 4 --crop_2 5 --crop_3 7




export CUDA_VISIBLE_DEVICES=6
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt.py --crop_1 37 --crop_2 40 --crop_3 41

export CUDA_VISIBLE_DEVICES=2
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt.py --g 0 --h 0 --crop_1 33 --crop_2 36 --crop_3 41

export CUDA_VISIBLE_DEVICES=3
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt.py --g 0 --h 0 --crop_1 34 --crop_2 37 --crop_3 40

crop_types_all_list = [ 1,  2,  3,|||  4,  5,  7,|||  8,  9, 10,||| 11, 12, 13,||| 14, 15, 16,||| 18, 19, 20,||| 21, 23, 27,||| 28, 30, 32, |||33, 34, 35, |||36, 37, 40, ||| 37, 40, 41]
vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}

(33, 34, 36, 37, 40, 41)
    chosen_crop_types_list_list = [[34, 36, 37], [33, 36, 40], [33, 36, 41]]


Winter crops
	•	WINTER_BARLEY (33)
	•	WINTER_OAT (34)
	•	WINTER_OTHER_CEREALS (35)
	•	WINTER_RAPESEED (36)
	•	WINTER_RYE (37)
	•	WINTER_TRITICALE (40)
	•	WINTER_WHEAT (41)


    in the landscape (g = 0, h = 0) there is no WINTER_OTHER_CEREALS (35)
    in the landscape (g = 0, h = 0) there is WINTER_OTHER_CEREALS (35) and all others

'''

# SIMILAR POPULATIONS 
# SPRING WHEAT, WINTER RYE, FLOWERING LEGUMES : 28, 37, 5

# well trained groups : [1, 2, 3], 
# Winter rapseed, winter barley, winter rye : 36, 33, 37 : 33, 36, 37
# winter wheat, winter barley, winter rapseed : 33, 36, 41
# winter triticle, alfalfa, legumes : 1, 11, 40
# winter triyicle, winter rye, winter other cereals : 40, 37, 35 : 35, 37, 40
# To get a sense for how much memory you have available to your processes you can run:
# cat /proc/meminfo | grep MemTotal
# sunflower, permnent plantations, soy : 


import segmentation_models_3D as sm
#import segmentation_models as sm


from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=8, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_X = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return batch_X, batch_y

    def on_epoch_end(self):
        # Shuffle the data after each epoch if specified
        if self.shuffle:
            indices = np.arange(len(self.X))
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]


class DataGenerator_upd(Sequence):
    def __init__(self, X, y, batch_size=8, shuffle=True, preprocess=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        if self.preprocess:
            batch_X = self.preprocess(batch_X)
        
        return batch_X, batch_y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)





from tensorflow.keras.callbacks import Callback

class SaveEveryNEpoch(Callback):
    def __init__(self, save_freq, model_save_path):
        super(SaveEveryNEpoch, self).__init__()
        self.save_freq = save_freq
        self.model_save_path = model_save_path

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            model_save_path = self.model_save_path.format(epoch=epoch + 1)
            self.model.save(model_save_path)
            print(f'\nModel saved at epoch {epoch + 1} to {model_save_path}')




num_epochs = 2000
vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}

import argparse

parser = argparse.ArgumentParser(description='Enter crop type numbers in order')
parser.add_argument('--crop_1', type=int, default=1, help='Select crop type')
parser.add_argument('--crop_2', type=int, default=2, help='Select crop type')
parser.add_argument('--crop_3', type=int, default=3, help='Select crop type')
parser.add_argument('--g', type=int, default=0, help='Select crop type')
parser.add_argument('--h', type=int, default=0, help='Select crop type')


args = parser.parse_args()

cr_1 = args.crop_1
cr_2 = args.crop_2
cr_3 = args.crop_3
g = args.g
h = args.h

chosen_crop_types_list = [cr_1, cr_2, cr_3]

chosen_crop_types_list1 = [cr_1, cr_2, cr_3,  4,  5]

crop_types_all_list = [ 0, 1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 41]


print("vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[1]]", vista_crop_dict[chosen_crop_types_list[0]], vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[2]])

sampling_group_fractions = [1.0, 1.0, 1.0]


#all_input_img = []
#all_input_mask = []

checkpoint_path = './storage/data_gen_model/checkpoints/3D_unet_g_'+str(g)+'_h_'+str(h)+'_labels_' + vista_crop_dict[chosen_crop_types_list[0]] + '_' + vista_crop_dict[chosen_crop_types_list[1]] + '_' + vista_crop_dict[chosen_crop_types_list[2]] + '_epoch_{epoch:02d}.h5'

save_every_50_epochs = SaveEveryNEpoch(save_freq=50, model_save_path=checkpoint_path)

all_input_img_f = []
all_input_mask_f = []
counted = 0
for crop_no in chosen_crop_types_list:
    chosen_crop_type = vista_crop_dict[crop_no]
    print("chosen_crop_type", chosen_crop_type)

    input_img_f = io.imread('./storage/per_crop_data_labels_g_'+str(g)+'_h_'+str(h)+'/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n.tif')[:1000]
    input_mask_f = io.imread('./storage/per_crop_data_labels_g_'+str(g)+'_h_'+str(h)+'/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n.tif').astype(np.uint8)[:1000]

    #input_img_f = tifffile.memmap('./storage/per_crop_data_labels_g_'+str(g)+'_h_'+str(h)+'/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n.tif').astype(np.float16)[:1000]
    #input_img_f = tifffile.memmap('./storage/per_crop_data_labels_g_'+str(g)+'_h_'+str(h)+'/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n.tif')[:1000]
    #input_mask_f = tifffile.memmap('./storage/per_crop_data_labels_g_'+str(g)+'_h_'+str(h)+'/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n.tif').astype(np.uint8)[:1000]

    drop = int(len(input_img_f)*sampling_group_fractions[counted]) - 2

    input_img_f = input_img_f[:drop]
    input_mask_f = input_mask_f[:drop]

    all_input_img_f.append(input_img_f)
    all_input_mask_f.append(input_mask_f)
    counted+=1    
all_input_img_f = np.concatenate((all_input_img_f), axis=0)
all_input_mask_f = np.concatenate((all_input_mask_f), axis=0)


print("all_input_img_f.shape", all_input_img_f.shape)
print("all_input_mask_f.shape", all_input_mask_f.shape)


input_img = all_input_img_f
input_mask = all_input_mask_f


print("1 input_img.max()", input_img.max())
print("1 input_img.min()", input_img.min())


print("2 input_img.max()", input_img.max())
print("2 input_img.min()", input_img.min())


del all_input_img_f
del all_input_mask_f
del input_img_f
del input_mask_f
gc.collect()


input_mask = np.repeat(input_mask[:, np.newaxis, :, :], repeats=64, axis=1)


print("final input_img.shape", input_img.shape)
print("final input_mask.shape", input_mask.shape)



mi = 0
for k in crop_types_all_list:
    if k in chosen_crop_types_list:
        input_mask[input_mask==k]=mi+1
        mi+=1
    else:
        input_mask[input_mask==k]=0

###########################################################################################################

unique_elements, element_counts = np.unique(input_mask, return_counts=True)

print("2 : unique_elements, element_counts", unique_elements, element_counts)

# selected_counts = np.array([element_counts[0], element_counts[np.where(unique_elements == chosen_crop_types_list[0])[0][0]], element_counts[np.where(unique_elements == chosen_crop_types_list[1])[0][0]], element_counts[np.where(unique_elements == chosen_crop_types_list[2])[0][0]]])

#selected_counts = np.array([element_counts[0], element_counts[np.where(unique_elements == 1)[0][0]], element_counts[np.where(unique_elements == 2)[0][0]], element_counts[np.where(unique_elements == 3)[0][0]]])

selected_counts = np.array(element_counts)

selected_counts_fractions = selected_counts/np.sum(selected_counts)
selected_counts_fractions_f = selected_counts_fractions**(-1)
weights = selected_counts_fractions_f/np.sum(selected_counts_fractions_f)

del selected_counts_fractions_f
del selected_counts_fractions
del selected_counts
del element_counts
del unique_elements
gc.collect()

print("weights", weights)
print("weights[0]", weights[0])
print("weights[1]", weights[1])
print("weights[2]", weights[2])
print("weights[3]", weights[3])

###########################################################################################################




lai_uniques = 0
n_classes=4



train_img = np.stack((input_img,)*3, axis=-1)
train_mask = np.expand_dims(input_mask, axis=4)
train_mask_cat = to_categorical(train_mask, num_classes=n_classes)

X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.10, random_state = 0)


print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)


del train_mask_cat
del train_mask
del train_img
del input_img
del input_mask
#del input_img_f 
#del input_mask0
'''del input_img1
del input_mask1
del input_img2
del input_mask2
del input_img3
del input_mask3
del input_img4
del input_mask4
del input_img5
del input_mask5
del input_img6
del input_mask6
del input_img7
del input_mask7
del input_img8
del input_mask8
del input_img9
del input_mask9
del input_img10
del input_mask10'''


'''train_img = 0
train_mask_cat = 0
train_mask = 0
input_mask = 0
input_img = 0'''
gc.collect()


def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)




encoder_weights = 'imagenet'
BACKBONE = 'vgg16'  #Try vgg16, efficientnetb7, inceptionv3, resnet50
activation = 'softmax'
patch_size = 64
n_classes = 4
channels=3

LR = 0.000001

#LR = 0.000001

optim = keras.optimizers.Adam(LR)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([weights[0], weights[1], weights[2], weights[3]])) 

focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


preprocess_input = sm.get_preprocessing(BACKBONE)

X_train_prep = preprocess_input(X_train)
X_test_prep = preprocess_input(X_test)

del X_train
del X_test
gc.collect()

model = sm.Unet(BACKBONE, classes=n_classes, 
                input_shape=(patch_size, patch_size, patch_size, channels), 
                encoder_weights=None,
                activation=activation)

model = sm.Unet(BACKBONE, classes=n_classes, 
                input_shape=(patch_size, patch_size, patch_size, channels), 
                encoder_weights=encoder_weights,
                activation=activation)

model.load_weights('/data1/chethan/vgg_storage/vgg16_inp_channel_3_tch_0_top_False.h5', by_name=True, skip_mismatch=True)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())

train_generator = DataGenerator(X_train_prep, y_train, batch_size=8)
validation_generator = DataGenerator(X_test_prep, y_test, batch_size=8)

#train_generator = DataGenerator_upd(X_train, y_train, batch_size=8, preprocess=preprocess_input)
#validation_generator = DataGenerator_upd(X_test, y_test, batch_size=8, preprocess=preprocess_input)

del X_train_prep
del y_train
del X_test_prep
del y_test
gc.collect()


history = model.fit(
    train_generator,
    epochs=num_epochs,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[save_every_50_epochs]  # Add the custom callback here
)




model.save('./storage/data_gen_model/3D_unet_g_'+str(g)+'_h_'+str(h)+'_labels_'+vista_crop_dict[chosen_crop_types_list[0]]+'_'+vista_crop_dict[chosen_crop_types_list[1]]+'_'+vista_crop_dict[chosen_crop_types_list[2]]+'_num_epocs_'+str(num_epochs)+'.h5')

