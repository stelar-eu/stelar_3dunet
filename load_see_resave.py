#from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file
import matplotlib.pyplot as plt
#import torch
import numpy as np
#datapath = './dataset/france/lai_ras/'
import random
import glob
import tifffile
import logging


from skimage import io



'''filepaths = glob.glob('/home/luser/UniBw-STELAR/dataset/france2/processed_lai_npy/*.npy')
filepaths.sort()'''

# put a universal seed for all random initialiazitations


'''

export CUDA_VISIBLE_DEVICES=0
cd stelar_3dunet/
conda activate inn
python tiff_saving.py

'''
import torch
#device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")




input_img1 = io.imread('/mdadm0/chethan_krishnamurth/stelar_3dunet/storage/per_crop_data_labels/CLOVER/trainCLOVERn2.tif')
input_mask1 = io.imread('/mdadm0/chethan_krishnamurth/stelar_3dunet/storage/per_crop_data_labels/CLOVER/labCLOVERn2.tif').astype(np.uint8)


print("input_img1.shape", input_img1.shape)

print("input_mask1.shape", input_mask1.shape)


tifffile.imsave('/mdadm0/chethan_krishnamurth/stelar_3dunet/storage/per_crop_data_labels/CLOVER/trainCLOVERn2_.tif', input_img1)
tifffile.imsave('/mdadm0/chethan_krishnamurth/stelar_3dunet/storage/per_crop_data_labels/CLOVER/labCLOVERn2_.tif', input_mask1)
