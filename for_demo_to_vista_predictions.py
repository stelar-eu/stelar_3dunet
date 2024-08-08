#from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file
import matplotlib.pyplot as plt
#import torch
import numpy as np
#datapath = './dataset/france/lai_ras/'
import random
import glob
import tifffile
filepaths = glob.glob('/home/luser/UniBw-STELAR/dataset/france2/processed_lai_npy/*.npy')
filepaths.sort()

# put a universal seed for all random initialiazitations


'''
export CUDA_VISIBLE_DEVICES=1
cd stelar_3d/
conda activate inn
python traiing_data_per_crop.py --chosen_crop_types 23

'''
import torch
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}



labels = np.load('/home/luser/stelar_3d/storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)

temporal_strip = torch.tensor([]).to(device)

filepaths = filepaths[50:64]

for filepath in filepaths:
    numpy_array = np.load(filepath)
    torch_tensor = torch.from_numpy(numpy_array).to(device)
    numpy_array = 0.0

    #print(torch_tensor[x_corner:x_corner+64, y_corner:y_corner+64].shape)
    #temporal_strip.append(torch_tensor[x_coord:x_coord+64, y_coord:y_coord+64])
    temporal_strip = torch.cat((temporal_strip, torch_tensor.unsqueeze(0)), axis=0)
    print("temporal_strip.shape", temporal_strip.shape)
#temporal_batches = torch.cat((temporal_batches, temporal_strip.unsqueeze(0)) , axis=0)


array_lai = temporal_strip.cpu().numpy()
tifffile.imsave('/home/luser/stelar_3d/storage/first64/50to64.tif', array_lai)


