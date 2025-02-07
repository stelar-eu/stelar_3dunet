#from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file

'''import torch
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
temporal_batches = torch.tensor([]).to(device)'''



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


filepaths = glob.glob('./dataset/france2/processed_lai_npy/*.npy')
filepaths.sort()

# put a universal seed for all random initialiazitations


'''

export CUDA_VISIBLE_DEVICES=1
cd stelar_3dunet/
conda activate inn2
python traiing_data_per_crop_test_set_create.py --chosen_crop_types 33

export CUDA_VISIBLE_DEVICES=1
cd stelar_3dunet/
conda activate inn2
python traiing_data_per_crop_test_set_create.py --chosen_crop_types 34

export CUDA_VISIBLE_DEVICES=1
cd stelar_3dunet/
conda activate inn2
python traiing_data_per_crop_test_set_create.py --chosen_crop_types 37

export CUDA_VISIBLE_DEVICES=1
cd stelar_3dunet/
conda activate inn2
python traiing_data_per_crop_test_set_create.py --chosen_crop_types 40

export CUDA_VISIBLE_DEVICES=1
cd stelar_3dunet/
conda activate inn2
python traiing_data_per_crop_test_set_create.py --chosen_crop_types 41

export CUDA_VISIBLE_DEVICES=1
cd stelar_3dunet/
conda activate inn2
python traiing_data_per_crop_test_set_create.py --chosen_crop_types 36



crop_types_all_list = [ 1,  2,  3,|||  4,  5,  7,|||  8,  9, 10,||| 11, 12, 13,||| 14, 15, 16,||| 18, 19, 20,||| 21, 23, 27,||| 28, 30, 32, |||33, 34, 35, |||36, 37, 40, ||| 37, 40, 41]




general time strips : time_strip_inds = [0, 52, 84, 116, 158, 190]

Winter crops
	•	WINTER_BARLEY (33)
	•	WINTER_OAT (34)
	•	WINTER_OTHER_CEREALS (35)
	•	WINTER_RAPESEED (36)
	•	WINTER_RYE (37)
	•	WINTER_TRITICALE (40)
	•	WINTER_WHEAT (41)

    winter time indices : 2020: 80 to 86
                        : 2021 : 0 to 21, 79 to 85
                        : 2022 : 0 to 23, 84 to 91

    combined list indices : 80 to 86, 87 to 108, 166 to 172, 173 to 196, 257 to 264

    time_strip_inds_winter = [85, 90, 95, 100, 105, 170, 175, 180, 185, 190, 195, 260]

    time_strip_inds_winter = [90, 100, 170, 180, 190, 260]

    
    in the landscape (g = 0, h = 0) there is no WINTER_OTHER_CEREALS (35)


'''
import torch



device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
# start from potato
vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}



import argparse

parser = argparse.ArgumentParser(description='Crop speicific LAI and labels sampling')
parser.add_argument('--chosen_crop_types', type=int, default=3, help='Select crop type')
#parser.add_argument('--randcon', type=int, default=3, help='Select rancon')

args = parser.parse_args()


chosen_crop_types = args.chosen_crop_types
#randcon = args.randcon

logging.info(f"Selected crop type: {vista_crop_dict[chosen_crop_types]}")

#chosen_crop_types = 1

print("vista_crop_dict[chosen_crop_types]", vista_crop_dict[chosen_crop_types])

per_crop_spatio_temporal_data_size = 20000

#labels_p = np.load('/home/luser/STELAR_Workbenches/saved_labels/lai_specific_france_mask_10_top.npy')
#labels = np.load('/home/luser/UniBw-STELAR/dataset/test_saves/vista_labes_image.npy').astype(np.uint8)
labels = np.load('./storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)


#draw_time_index = random.randint(0, 245)
#selcted_time_files = filepaths[draw_time_index:draw_time_index+20]
#big_stack = torch.tensor([]).to(device)

#for k in range(100):

    #x_ind = random.choices(range(14), k=64)
    #y_ind = random.choices(range(14), k=64)


#time_strip_inds = [0, 52, 84, 116, 158, 190]
#time_strip_inds = [90, 100, 170, 178, 185, 190]
time_strip_inds = [60, 146, 195, 60, 146, 195]




all_lai = torch.tensor([]).to(device)
all_labels = torch.tensor([]).to(device)


for time_strip_ind in time_strip_inds:

    temporal_batches = torch.tensor([]).to(device)
    spatial_label_batches = torch.tensor([]).to(device)

    print("len(filepaths)", len(filepaths))

    #time_strip_ind =  np.random.randint(0, len(filepaths)-65)

    print("time_strip_ind", time_strip_ind)

    considered_filepaths = filepaths[time_strip_ind:time_strip_ind+64]

    for g in range(1):
        for h in range(1):
            temporal_strip = torch.tensor([]).to(device)

            for filepath in considered_filepaths:
                numpy_array = np.load(filepath)
                torch_tensor = torch.from_numpy(numpy_array).to(device)
                numpy_array = 0.0
                temporal_strip = torch.cat((temporal_strip, torch_tensor[5000*g:5000*g+5000, 5000*h:5000*h+5000].unsqueeze(0)), axis=0)

            labels_ = labels[5000*g:5000*g+5000, 5000*h:5000*h+5000]
            x_inds, y_inds = np.where(labels_==chosen_crop_types)

            print("(len(x_inds)!=0 and len(y_inds)!=0)", (len(x_inds)!=0 and len(y_inds)!=0))

            if(len(x_inds)!=0 and len(y_inds)!=0):

                print("len(x_inds)", len(x_inds))
                print("len(y_inds)", len(y_inds))

                '''print("temporal_strip.shape", temporal_strip.shape)
                print("labels.shape", labels.shape)'''

                for i in range(200):

                    random_corner = random.choices(range(len(x_inds)-2), k=1)[0]
                    x_corner = x_inds[random_corner]
                    y_corner = y_inds[random_corner]

                    if(x_corner>labels_.shape[0]-90):
                        x_corner = x_corner-90
                    if(y_corner>labels_.shape[1]-90):
                        y_corner = y_corner-90

                    '''print("labels_.shape", labels_.shape)
                    print("temporal_strip.shape", temporal_strip.shape)'''

                    space_label = labels_[x_corner:x_corner+64, y_corner:y_corner+64]
                    space_label = torch.from_numpy(space_label).to(device)

                    temporal_strip_ = temporal_strip[:, x_corner:x_corner+64, y_corner:y_corner+64]

                    '''print("space_label.shape", space_label.shape)
                    print("3 temporal_strip.shape", temporal_strip_.shape)'''

                    spatial_label_batches = torch.cat((spatial_label_batches, space_label.unsqueeze(0)), axis=0)

                    temporal_batches = torch.cat((temporal_batches, temporal_strip_.unsqueeze(0)), axis=0)

            else:    
                temporal_strip = torch.tensor([]).to(device)

    print("spatial_label_batches.shape", spatial_label_batches.shape)   
    print("temporal_batches.shape", temporal_batches.shape)    

    all_lai = torch.cat((all_lai, temporal_batches), axis=0)
    all_labels = torch.cat((all_labels, spatial_label_batches), axis=0) 


    print("all_lai.shape", all_lai.shape)   
    print("all_labels.shape", all_labels.shape)    


array_lai = all_lai.cpu().numpy()

print("Last all_lai.shape", all_lai.shape)   
print("Last all_labels.shape", all_labels.shape)    

tifffile.imsave('./storage/per_crop_data_labels_test/'+vista_crop_dict[chosen_crop_types]+'/train'+vista_crop_dict[chosen_crop_types]+'n.tif', array_lai)


array_lab = all_labels.cpu().numpy()
tifffile.imsave('./storage/per_crop_data_labels_test/'+vista_crop_dict[chosen_crop_types]+'/lab'+vista_crop_dict[chosen_crop_types]+'n.tif', array_lab)


'''spatial_label_batches = torch.cat((spatial_label_batches, space_label.unsqueeze(0)), axis=0)


print("temporal_batches.shape", temporal_batches.shape)
print("spatial_label_batches.shape", spatial_label_batches.shape)

array_lai = temporal_batches.cpu().numpy()


tifffile.imsave('./storage/per_crop_data_labels/'+vista_crop_dict[chosen_crop_types]+'/train'+vista_crop_dict[chosen_crop_types]+'n'+str(randcon)+'.tif', array_lai)


array_lab = spatial_label_batches.cpu().numpy()
tifffile.imsave('./storage/per_crop_data_labels/'+vista_crop_dict[chosen_crop_types]+'/lab'+vista_crop_dict[chosen_crop_types]+'n'+str(randcon)+'.tif', array_lab)

if(i%10==0):
    temporal_batches = torch.tensor([]).to(device)
    spatial_label_batches = torch.tensor([]).to(device)
    randcon +=1'''


