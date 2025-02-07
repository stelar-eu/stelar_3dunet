#from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file
import matplotlib.pyplot as plt
#import torch
import numpy as np
#datapath = './dataset/france/lai_ras/'
import random
import glob
import tifffile
import logging



# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


filepaths = glob.glob('/home/luser/UniBw-STELAR/dataset/france2/processed_lai_npy/*.npy')
filepaths.sort()

# put a universal seed for all random initialiazitations


'''

export CUDA_VISIBLE_DEVICES=0
cd stelar_3dunet/
conda activate inn
python tiff_saving.py

'''
import torch
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")





#labels_p = np.load('/home/luser/STELAR_Workbenches/saved_labels/lai_specific_france_mask_10_top.npy')
#labels = np.load('/home/luser/UniBw-STELAR/dataset/test_saves/vista_labes_image.npy').astype(np.uint8)
labels = np.load('/home/luser/stelar_3dunet/storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)

for filepath in filepaths:
    array_lai = np.load(filepath)
    array_lai = array_lai.astype(np.float32)


    print("array_lai.dtype", array_lai.dtype)

    print("full filepath", filepath)
    print("required filepath", filepath[-25:-4])
    #array_lai = temporal_batches.cpu().numpy()
    tifffile.imsave('/home/luser/UniBw-STELAR/dataset/france2/processed_lai_npy_tiff/'+filepath[-25:-4]+'.tif', array_lai)


