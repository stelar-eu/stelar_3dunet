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

    

export CUDA_VISIBLE_DEVICES=1
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi.py --g 3 --h 3 --crop_1 33 --crop_2 36 --crop_3 41

export CUDA_VISIBLE_DEVICES=2
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi.py --g 3 --h 3 --crop_1 34 --crop_2 37 --crop_3 40



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

export CUDA_VISIBLE_DEVICES=3
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi.py --g 1 --h 1 --crop_1 2 --crop_2 15 --crop_3 20 --fine_tune True


export CUDA_VISIBLE_DEVICES=4
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi.py --g 1 --h 1 --crop_1 21 --crop_2 23 --crop_3 28 --fine_tune True


summer_autumn crops

	• GRAIN_MAIZE(8)
    • GRASSLAND(9)
    • SUNFLOWER(30)

    • FOREST(7)
    • SILAGE_MAIZE(18)
    • SOY(19)


    
export CUDA_VISIBLE_DEVICES=3
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi.py --g 1 --h 1 --crop_1 8 --crop_2 9 --crop_3 30


export CUDA_VISIBLE_DEVICES=4
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi.py --g 1 --h 1 --crop_1 7 --crop_2 18 --crop_3 19



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

export CUDA_VISIBLE_DEVICES=5
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 3D_unet_data_generator_check_pt_data_effi.py --g 1 --h 1 --crop_1 4 --crop_2 7 --crop_3 9


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

parser.add_argument('--fine_tune', type=bool, default=False)
parser.add_argument('--cl_weights', type=bool, default=False)


args = parser.parse_args()

g, h = args.g, args.h
chosen_crop_types = [args.crop_1, args.crop_2, args.crop_3]
fine_tune = args.fine_tune
cl_weights = args.cl_weights



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

def save_3d_unet_samples(input_img, input_mask, crop_name, out_dir):
    os.makedirs(f"{out_dir}/images/{crop_name}/", exist_ok=True)
    os.makedirs(f"{out_dir}/masks/{crop_name}/", exist_ok=True)
    for i in tqdm(range(len(input_img)), desc=f"Saving {crop_name}"):
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

        mi = 0
        for k in crop_types_all_list:
            if k in chosen_crop_types:
                mask_3d[mask_3d==k]=mi+1
                mi+=1
            else:
                mask_3d[mask_3d==k]=0

        #print("3 mask_3d.shape", mask_3d.shape)

        '''plt.imshow(mask_3d[0])
        plt.savefig('/data1/chethan/stelar_3dunet/general_tests/savec'+str(i)+'.png')
        plt.close()'''

        imsave(f"{out_dir}/images/{crop_name}/img_{i}.tif", img.astype('float32'))
        imsave(f"{out_dir}/masks/{crop_name}/mask_{i}.tif", mask_3d.astype('uint8'))

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
'''for crop_id in chosen_crop_types:
    crop_name = vista_crop_dict[crop_id]

    input_img_list = []
    input_mask_list = []
    for g in range(2):
        for h in range(3):
            img_path = f"{BASE_DATA_DIR.format(g=g, h=h)}/{crop_name}/train{crop_name}n.tif"
            mask_path = f"{BASE_DATA_DIR.format(g=g, h=h)}/{crop_name}/lab{crop_name}n.tif"
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
    save_3d_unet_samples(stacked_img, stacked_mask, crop_name, PATCH_SAVE_DIR.format(g=g, h=h))'''


###########################################################################################################

g=4
h=4



# -----------------------------
# Load file paths & generators
# -----------------------------
#image_paths = sorted(glob(f"{PATCH_SAVE_DIR.format(g=g, h=h)}/images/*/*.tif"))
#mask_paths = sorted(glob(f"{PATCH_SAVE_DIR.format(g=g, h=h)}/masks/*/*.tif"))


image_paths = []
mask_paths = []

for crop_id in chosen_crop_types:
    crop_name = vista_crop_dict[crop_id]
    image_paths.extend(sorted(glob(f"{PATCH_SAVE_DIR.format(g=g, h=h)}/images/{crop_name}/*.tif")))
    mask_paths.extend(sorted(glob(f"{PATCH_SAVE_DIR.format(g=g, h=h)}/masks/{crop_name}/*.tif")))


X_train, X_val, y_train, y_val = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=42)

train_gen = Lazy3DUNetGenerator(X_train, y_train)
val_gen = Lazy3DUNetGenerator(X_val, y_val)

# -----------------------------
# Build and train model
# -----------------------------
BACKBONE = 'vgg16'
model = sm.Unet(BACKBONE, input_shape=(64, 64, 64, 3), classes=4, encoder_weights=None, activation='softmax')

#if fine_tune:
model.load_weights(PRETRAINED_PATH.format(g=g, h=h, c1=args.crop_1, c2=args.crop_2, c3=args.crop_3), by_name=True, skip_mismatch=True)
#else:
    
#model.load_weights(VGG_WEIGHT_PATH, by_name=True, skip_mismatch=True)


'''if args.crop_1==33 and  args.crop_2==36 and args.crop_3==41:
    weights = [0.02053242, 0.489391, 0.36363955, 0.12643702]
    print("this is true")

if args.crop_1==34 and  args.crop_2==37 and args.crop_3==40:
    weights = [0.00634919, 0.28517157, 0.47803394, 0.2304453]
    print("this is true")

if args.crop_1==8 and  args.crop_2==9 and args.crop_3==30:
    weights = [0.04375175, 0.2613057, 0.27038112, 0.42456143]
    print("this is true")

if args.crop_1==7 and  args.crop_2==18 and args.crop_3==19:
    weights = [0.01299109, 0.48659389, 0.27530874, 0.22510628]
    print("this is true")

if cl_weights:
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([weights[0], weights[1], weights[2], weights[3]])) 
else:
    dice_loss = sm.losses.DiceLoss() '''


#focal_loss = sm.losses.CategoricalFocalLoss()
#total_loss = dice_loss + (1 * focal_loss)


'''model.compile(
    optimizer=Adam(1e-6),
    loss=total_loss,
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
)'''


model.compile(
    optimizer=Adam(1e-6),
    loss=sm.losses.DiceLoss() + sm.losses.CategoricalFocalLoss(),
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
)


class SaveEveryNEpoch(tf.keras.callbacks.Callback):
    def __init__(self, freq, path): self.freq = freq; self.path = path
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            self.model.save(self.path.format(epoch=epoch+1))
            print(f"✅ Model saved at epoch {epoch+1}")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2000,
    callbacks=[SaveEveryNEpoch(10, CHECKPOINT_PATH.format(g=g, h=h, c1=args.crop_1, c2=args.crop_2, c3=args.crop_3))],
    verbose=1
)

# Final model save
model.save(f"./models/final_3D_unet_g{g}_h{h}_crop_{args.crop_1}_{args.crop_2}_{args.crop_3}.h5")