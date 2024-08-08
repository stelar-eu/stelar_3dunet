
import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)


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

export CUDA_VISIBLE_DEVICES=1
conda deactivate
conda deactivate
cd stelar_3dunet/
source spt19/bin/activate
python3 3D_unet_bias_mit_1_representation_bias_mitigation_through_class_weights_corrected_more_data.py --crop_1 18 --crop_2 19 --crop_3 20
python3 3D_unet_bias_mit_1_representation_bias_mitigation_through_class_weights_corrected_more_data.py --crop_1 14 --crop_2 15 --crop_3 16


export CUDA_VISIBLE_DEVICES=1
conda deactivate
conda deactivate
cd stelar_3dunet/
source spt19/bin/activate
python3 3D_unet_bias_mit_1_representation_bias_mitigation_through_class_weights_corrected_more_data.py --crop_1 4 --crop_2 5 --crop_3 7
python3 3D_unet_bias_mit_1_representation_bias_mitigation_through_class_weights_corrected_more_data.py --crop_1 33 --crop_2 34 --crop_3 35
python3 3D_unet_bias_mit_1_representation_bias_mitigation_through_class_weights_corrected_more_data.py --crop_1 36 --crop_2 37 --crop_3 40
python3 3D_unet_bias_mit_1_representation_bias_mitigation_through_class_weights_corrected_more_data.py --crop_1 37 --crop_2 40 --crop_3 41


****** memory error issues *************

export CUDA_VISIBLE_DEVICES=1
conda deactivate
conda deactivate
cd stelar_3dunet/
source spt19/bin/activate
export TF_GPU_ALLOCATOR=cuda_malloc_async
python3 3D_unet_bias_mit_1_representation_bias_mitigation_through_class_weights_corrected_more_data.py --crop_1 1 --crop_2 2 --crop_3 3


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
    input_img = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train_'+vista_crop_dict[crop_no]+'.tif')
    input_mask = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab_'+vista_crop_dict[crop_no]+'.tif').astype(np.uint8)

    print("input_img.shape", input_img.shape)
    print("input_mask.shape", input_mask.shape)

    input_img0 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n0.tif')
    input_mask0 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n0.tif').astype(np.uint8)
    print("input_img0.shape", input_img0.shape)
    print("input_mask0.shape", input_mask0.shape)


    input_img1 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n1.tif')
    input_mask1 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n1.tif').astype(np.uint8)
    print("input_img1.shape", input_img1.shape)
    print("input_mask1.shape", input_mask1.shape)

    input_img2 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n2.tif')
    input_mask2 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n2.tif').astype(np.uint8)
    print("input_img2.shape", input_img2.shape)
    print("input_mask2.shape", input_mask2.shape)

    input_img3 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n3.tif')
    input_mask3 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n3.tif').astype(np.uint8)
    print("input_img3.shape", input_img3.shape)
    print("input_mask3.shape", input_mask3.shape)

    input_img4 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n4.tif')
    input_mask4 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n4.tif').astype(np.uint8)
    print("input_img4.shape", input_img4.shape)
    print("input_mask4.shape", input_mask4.shape)

    input_img5 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n5.tif')
    input_mask5 = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n5.tif').astype(np.uint8)
    print("input_img5.shape", input_img5.shape)
    print("input_mask5.shape", input_mask5.shape)

    input_img_f = np.concatenate((input_img, input_img0, input_img1, input_img2, input_img3, input_img4, input_img5), axis=0)
    input_mask_f = np.concatenate((input_mask, input_mask0, input_mask1, input_mask2, input_mask3, input_mask4, input_mask5), axis=0)


    bis = int(len(input_img_f)*sampling_group_fractions[counted]) - 50

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

input_img = np.concatenate((all_input_img_f), axis=0).reshape(-1, 64, 64, 64)
#input_img = np.array(all_input_img).reshape(-1, 64, 64, 64)
#input_mask = np.array(all_input_mask).reshape(-1, 64, 64)
input_mask = np.concatenate((all_input_mask_f), axis=0).reshape(-1, 64, 64)

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


input_mask[input_mask<chosen_crop_types_list[0]]=0
input_mask[input_mask>chosen_crop_types_list[-1]]=0
for i in range(3):
    input_mask[input_mask==chosen_crop_types_list[i]]=i+1

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


print("weights", weights)
print("weights[0]", weights[0])
print("weights[1]", weights[1])
print("weights[2]", weights[2])
print("weights[3]", weights[3])

###########################################################################################################





#input_mask = np.repeat(input_mask[:, np.newaxis, :, :], repeats=64, axis=1)

#input_img[input_img==0] = np.median(input_img)

'''for m in range(64):
    for i in range(64):
        for j in range(64):
            input_img[m,:,i,j][input_img[m,:,i,j]==0] = np.median(input_img[m,:,i,j]).astype(np.uint8)'''

lai_uniques = np.unique(input_img)

for n in range(len(lai_uniques)):
  input_img[input_img==lai_uniques[n]]=n
input_img = input_img.astype(np.uint8)


lai_uniques = 0
n_classes=4



train_img = np.stack((input_img,)*3, axis=-1)
train_mask = np.expand_dims(input_mask, axis=4)
train_mask_cat = to_categorical(train_mask, num_classes=n_classes)

X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.10, random_state = 0)

X_train = X_train[:9000]
y_train = y_train[:9000]
X_test = X_test[:1000]
y_test = y_test[:1000]

print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)


del train_mask_cat
del train_mask
del train_img
del input_img
del input_mask
del input_img0 
del input_mask0
del input_img1
del input_mask1
del input_img2
del input_mask2
del input_img3
del input_mask3
del input_img4
del input_mask4
del input_img5
del input_mask5


'''train_img = 0
train_mask_cat = 0
train_mask = 0
input_mask = 0
input_img = 0'''



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

LR = 0.0001

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


#Define the model. Here we use Unet but we can also use other model architectures from the library.
model = sm.Unet(BACKBONE, classes=n_classes, 
                input_shape=(patch_size, patch_size, patch_size, channels), 
                encoder_weights=encoder_weights,
                activation=activation)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())


print("Did fitting start ?")
print("X_train_prep.shape", X_train_prep.shape)
#Fit the model
history=model.fit(X_train_prep, 
          y_train,
          batch_size=8, 
          epochs=num_epochs,
          verbose=1,
          validation_data=(X_test_prep, y_test))


model.save('/home/luser/stelar_3dunet/storage/saved_model_bias_miti_class_weights_corrected_more_data/3D_unet_res_labels_'+vista_crop_dict[chosen_crop_types_list[0]]+'_'+vista_crop_dict[chosen_crop_types_list[1]]+'_'+vista_crop_dict[chosen_crop_types_list[2]]+'_num_epocs_'+str(num_epochs)+'.h5')
