import os
import argparse
import numpy as np
from skimage import io
from tifffile import imsave
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence, to_categorical
import segmentation_models_3D as sm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from segmentation_models_3D import get_preprocessing
preprocess_input = get_preprocessing('vgg16')

import os
os.environ["TMPDIR"] = "/data1/chethan/tmp-tf"

'''
Winter crops
	•	WINTER_BARLEY (33)
	•	WINTER_OAT (34)
	•	WINTER_OTHER_CEREALS (35)
	•	WINTER_RAPESEED (36)
	•	WINTER_RYE (37)
	•	WINTER_TRITICALE (40)
	•	WINTER_WHEAT (41)
    chosen_crop_types_list_list = [[33, 36, 41], [34, 37, 40]]

    in the landscape (g = 0, h = 0) there is no WINTER_OTHER_CEREALS (35)

    

export CUDA_VISIBLE_DEVICES=5
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi_prestep.py --g 3 --h 3 --crop_1 33 --crop_2 36 --crop_3 41 --season winter

export CUDA_VISIBLE_DEVICES=2
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi_prestep.py --g 3 --h 3 --crop_1 34 --crop_2 37 --crop_3 40 --season winter



Spring crops
	•	BEET(2)
	•	POTATO(15)
    •	SPRING_BARLEY(20)    
	•	SPRING_OAT(21)
	•	SPRING_RAPESEED(23)
	•	SPRING_WHEAT(28)

    chosen_crop_types_list_list = [[2, 15, 20], [21, 23, 28]]
    spring time indices : 2020: 80 to 86
                        : 2021 : 0 to 21, 79 to 85
                        : 2022 : 0 to 23, 84 to 91
time_strip_inds = [20, 30, 120, 130, 175, 180]

export CUDA_VISIBLE_DEVICES=5
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi_prestep.py --g 1 --h 1 --crop_1 2 --crop_2 15 --crop_3 20 --season spring


export CUDA_VISIBLE_DEVICES=4
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi_prestep.py --g 1 --h 1 --crop_1 21 --crop_2 23 --crop_3 28 --season spring


summer_autumn crops

	• GRAIN_MAIZE(8)
    • GRASSLAND(9)
    • SUNFLOWER(30)

    • FOREST(7)
    • SILAGE_MAIZE(18)
    • SOY(19)


    
export CUDA_VISIBLE_DEVICES=7
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi_prestep.py --g 1 --h 1 --crop_1 8 --crop_2 9 --crop_3 30 --season summer_autumn


export CUDA_VISIBLE_DEVICES=4
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi_prestep.py --g 1 --h 1 --crop_1 7 --crop_2 18 --crop_3 19 --season summer_autumn



#########clearing space -> removing data

find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/BEET -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/POTATO -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/SPRING_BARLEY -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/SPRING_OAT -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/SPRING_RAPESEED -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/SPRING_WHEAT -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/WINTER_BARLEY -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/WINTER_OAT -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/WINTER_RYE -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/WINTER_TRITICALE -type f -delete
find /data1/chethan/stelar_3dunet/storage/patches/g3_h3/images/WINTER_WHEAT -type f -delete



winter_spring_summer 

FLAX((4)
FOREST(7)
GRASSLAND(9)

export CUDA_VISIBLE_DEVICES=6
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi_prestep.py --g 1 --h 1 --crop_1 4 --crop_2 7 --crop_3 9 --season winter_spring_summer


'''



# -----------------------------
# Configurable paths (ONLY EDIT THESE)
# -----------------------------
BASE_DATA_DIR = './storage/per_crop_data_labels_g_{g}_h_{h}'
PATCH_SAVE_DIR = './storage/patches/g{g}_h{h}'  # where patches will be saved
VGG_WEIGHT_PATH = '/data1/chethan/vgg_storage/vgg16_inp_channel_3_tch_0_top_False.h5'
CHECKPOINT_PATH = './checkpoints/3D_unet_g{g}_h{h}_crop_{c1}_{c2}_{c3}_epoch_{{epoch:02d}}.h5'
PRETRAINED_PATH = './checkpoints_f1/3D_unet_g{g}_h{h}_crop_{c1}_{c2}_{c3}_epoch_f.h5'


# -----------------------------
# Argparse setup
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--g', type=int, default=0)
parser.add_argument('--h', type=int, default=0)
parser.add_argument('--crop_1', type=int, default=33)
parser.add_argument('--crop_2', type=int, default=36)
parser.add_argument('--crop_3', type=int, default=41)

parser.add_argument('--season', type=str, default=0, help='input season')


parser.add_argument('--fine_tune', type=bool, default=False)
parser.add_argument('--cl_weights', type=bool, default=False)


args = parser.parse_args()

g, h = args.g, args.h
chosen_crop_types = [args.crop_1, args.crop_2, args.crop_3]
fine_tune = args.fine_tune
cl_weights = args.cl_weights
season = args.season



vista_crop_dict = {
    0:'NA', 1:'ALFALFA', 2:'BEET', 3:'CLOVER', 4:'FLAX', 5:'FLOWERING_LEGUMES',
    6:'FLOWERS', 7:'FOREST', 8:'GRAIN_MAIZE', 9:'GRASSLAND', 10:'HOPS', 11:'LEGUMES',
    12:'VISTA_NA', 13:'PERMANENT_PLANTATIONS', 14:'PLASTIC', 15:'POTATO', 16:'PUMPKIN',
    17:'RICE', 18:'SILAGE_MAIZE', 19:'SOY', 20:'SPRING_BARLEY', 21:'SPRING_OAT',
    22:'SPRING_OTHER_CEREALS', 23:'SPRING_RAPESEED', 24:'SPRING_RYE', 25:'SPRING_SORGHUM',
    26:'SPRING_SPELT', 27:'SPRING_TRITICALE', 28:'SPRING_WHEAT', 29:'SUGARBEET',
    30:'SUNFLOWER', 31:'SWEET_POTATOES', 32:'TEMPORARY_GRASSLAND', 33:'WINTER_BARLEY',
    34:'WINTER_OAT', 35:'WINTER_OTHER_CEREALS', 36:'WINTER_RAPESEED', 37:'WINTER_RYE',
    38:'WINTER_SORGHUM', 39:'WINTER_SPELT', 40:'WINTER_TRITICALE', 41:'WINTER_WHEAT'
}

crop_types_all_list = [ 0, 1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 41]


# -----------------------------
# Patch Saver
# -----------------------------
from matplotlib import pyplot as plt

def save_3d_unet_samples(input_img, input_mask, season, out_dir):
    os.makedirs(f"{out_dir}/images/{season}/", exist_ok=True)
    os.makedirs(f"{out_dir}/masks/{season}/", exist_ok=True)
    for i in tqdm(range(len(input_img)), desc=f"Saving {season}"):
        img = input_img[i]
        mask = input_mask[i]

        '''print("1 mask.shape", mask.shape)
        plt.imshow(mask)
        plt.savefig('/data1/chethan/stelar_3dunet/general_tests/saveb'+str(i)+'.png')
        plt.close()'''
        mask_3d = np.repeat(mask[np.newaxis, :, :], repeats=64, axis=0)   # here there seems to be a problem
        #input_mask = np.repeat(input_mask[:, np.newaxis, :, :], repeats=64, axis=1)

        '''plt.imshow(mask_3d[0])
        plt.savefig('/data1/chethan/stelar_3dunet/general_tests/savea'+str(i)+'.png')
        plt.close()'''

        #print("2 mask_3d.shape", mask_3d.shape)

        '''mi = 0
        for k in crop_types_all_list:
            if k in chosen_crop_types:
                mask_3d[mask_3d==k]=mi+1
                mi+=1
            else:
                mask_3d[mask_3d==k]=0'''

        #print("3 mask_3d.shape", mask_3d.shape)

        '''plt.imshow(mask_3d[0])
        plt.savefig('/data1/chethan/stelar_3dunet/general_tests/savec'+str(i)+'.png')
        plt.close()'''

        imsave(f"{out_dir}/images/{season}/img_{i}.tif", img.astype('float32'))
        imsave(f"{out_dir}/masks/{season}/mask_{i}.tif", mask_3d.astype('uint8'))

# -----------------------------
# Lazy Data Generator
# -----------------------------
class Lazy3DUNetGenerator(Sequence):
    def __init__(self, img_paths, mask_paths, batch_size=8, num_classes=4, shuffle=True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, index):
        idxs = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch, y_batch = [], []
        for i in idxs:
            img = io.imread(self.img_paths[i]).astype(np.float32)
            mask = io.imread(self.mask_paths[i]).astype(np.uint8)

            img = np.stack((img,) * 3, axis=-1)
            #img = preprocess_input(img)
            mask = np.expand_dims(mask, axis=-1)

            mask_cat = to_categorical(mask, num_classes=self.num_classes)
            X_batch.append(img)
            y_batch.append(mask_cat)

        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
            self.img_paths = [self.img_paths[i] for i in self.indices]
            self.mask_paths = [self.mask_paths[i] for i in self.indices]

# -----------------------------
# Save patches for selected crops
# -----------------------------
#for crop_id in chosen_crop_types:
    #crop_name = vista_crop_dict[crop_id]

input_img_list = []
input_mask_list = []
#for g in range(2):
for mm in range(3):
    img_path = f"/data1/chethan/stelar_3dunet/storage/{season}_general/lai_{season}_general{mm}.tif"
    mask_path = f"/data1/chethan/stelar_3dunet/storage/{season}_general/lab_{season}_general{mm}.tif"
    #input_img = io.imread(img_path).astype(np.float16)[:9000]
    #input_mask = io.imread(mask_path).astype(np.uint8)[:9000]
    print("whats happening")
    print("img_path", img_path)
    print("mask_path", mask_path)
    input_img = io.imread(img_path).astype(np.float32)
    input_mask = io.imread(mask_path).astype(np.uint8)
    print("input_img.shape", input_img.shape)
    print("input_img.shape", input_img.shape)
    print("g, h", g, h)

    input_img_list.append(input_img)
    input_mask_list.append(input_mask)
# Stack all loaded tiles
stacked_img = np.concatenate(input_img_list, axis=0)  # or np.stack if they have the same shape
stacked_mask = np.concatenate(input_mask_list, axis=0)

print("Final stacked_img.shape:", stacked_img.shape)
print("Final stacked_mask.shape:", stacked_mask.shape)

g=4
h=4
save_3d_unet_samples(stacked_img, stacked_mask, season, PATCH_SAVE_DIR.format(g=g, h=h))


