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

export CUDA_VISIBLE_DEVICES=1
cd stelar_3dunet/
conda activate /home/luser/anaconda3/envs/inn
python traiing_data_per_crop.py --chosen_crop_types 10


start from  including 


crop_types_all_list = [ 1,  2,  3,|||  4,  5,  7,|||  8,  9, 10,||| 11, 12, 13,||| 14, 15, 16,||| 18, 19, 20,||| 21, 23, 27,||| 28, 30, 32, |||33, 34, 35, |||36, 37, 40, ||| 37, 40, 41]



'''
import torch
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
# start from potato
vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}



import argparse

parser = argparse.ArgumentParser(description='Crop speicific LAI and labels sampling')
parser.add_argument('--chosen_crop_types', type=int, default=3, help='Select crop type')
args = parser.parse_args()


chosen_crop_types = args.chosen_crop_types

logging.info(f"Selected crop type: {vista_crop_dict[chosen_crop_types]}")

#chosen_crop_types = 1

print("vista_crop_dict[chosen_crop_types]", vista_crop_dict[chosen_crop_types])

per_crop_spatio_temporal_data_size = 1000

#labels_p = np.load('/home/luser/STELAR_Workbenches/saved_labels/lai_specific_france_mask_10_top.npy')
#labels = np.load('/home/luser/UniBw-STELAR/dataset/test_saves/vista_labes_image.npy').astype(np.uint8)
labels = np.load('/home/luser/stelar_3dunet/storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)
x_inds, y_inds = np.where(labels==chosen_crop_types)

print("len(x_inds)", len(x_inds))
#draw_time_index = random.randint(0, 245)
#selcted_time_files = filepaths[draw_time_index:draw_time_index+20]
#big_stack = torch.tensor([]).to(device)

#for k in range(100):

    #x_ind = random.choices(range(14), k=64)
    #y_ind = random.choices(range(14), k=64)

temporal_batches = torch.tensor([]).to(device)
spatial_label_batches = torch.tensor([]).to(device)
for i in range(per_crop_spatio_temporal_data_size):
    time_strip_ind =  np.random.randint(0, len(filepaths)-65)
    considered_filepaths = filepaths[time_strip_ind:time_strip_ind+64]   
    temporal_strip = torch.tensor([]).to(device)

    #x_coord = x_ind[i] * 64
    #y_coord = y_ind[i] * 64
    
    random_corner = random.choices(range(len(x_inds)-2), k=1)[0]
    x_corner = x_inds[random_corner]
    y_corner = y_inds[random_corner]

    if(x_corner>labels.shape[0]-90):
        x_corner = x_corner-90
    if(y_corner>labels.shape[1]-90):
        y_corner = y_corner-90

    space_label = labels[x_corner:x_corner+64, y_corner:y_corner+64]
    space_label = torch.from_numpy(space_label).to(device)

    spatial_label_batches = torch.cat((spatial_label_batches, space_label.unsqueeze(0)), axis=0)
    for filepath in considered_filepaths:
        numpy_array = np.load(filepath)
        torch_tensor = torch.from_numpy(numpy_array).to(device)
        numpy_array = 0.0

        #print(torch_tensor[x_corner:x_corner+64, y_corner:y_corner+64].shape)
        #temporal_strip.append(torch_tensor[x_coord:x_coord+64, y_coord:y_coord+64])
        temporal_strip = torch.cat((temporal_strip, torch_tensor[x_corner:x_corner+64, y_corner:y_corner+64].unsqueeze(0)), axis=0)
        #print("temporal_strip.shape", temporal_strip.shape)
    temporal_batches = torch.cat((temporal_batches, temporal_strip.unsqueeze(0)) , axis=0)

    print("temporal_batches.shape", temporal_batches.shape)
    print("spatial_label_batches.shape", spatial_label_batches.shape)

    array_lai = temporal_batches.cpu().numpy()
    tifffile.imsave('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[chosen_crop_types]+'/train'+vista_crop_dict[chosen_crop_types]+'n10.tif', array_lai)


    array_lab = spatial_label_batches.cpu().numpy()
    tifffile.imsave('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[chosen_crop_types]+'/lab'+vista_crop_dict[chosen_crop_types]+'n10.tif', array_lab)
