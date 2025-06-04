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
conda activate dt2
python traiing_data_per_crop_test_set_create.py --g 1 --h 1  --chosen_crop_types 33
python traiing_data_per_crop_test_set_create.py --g 1 --h 0  --chosen_crop_types 34
python traiing_data_per_crop_test_set_create.py --g 1 --h 0  --chosen_crop_types 35
python traiing_data_per_crop_test_set_create.py --g 1 --h 0  --chosen_crop_types 36
python traiing_data_per_crop_test_set_create.py --g 1 --h 0  --chosen_crop_types 37
python traiing_data_per_crop_test_set_create.py --g 1 --h 0  --chosen_crop_types 40
python traiing_data_per_crop_test_set_create.py --g 1 --h 0  --chosen_crop_types 41


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


export CUDA_VISIBLE_DEVICES=6
cd stelar_3dunet/
conda activate dt2
python traiing_data_per_crop_test_set_create.py --g 1 --h 1  --chosen_crop_types 2
python traiing_data_per_crop_test_set_create.py --g 1 --h 1  --chosen_crop_types 15
python traiing_data_per_crop_test_set_create.py --g 1 --h 1  --chosen_crop_types 20
python traiing_data_per_crop_test_set_create.py --g 1 --h 1  --chosen_crop_types 21
python traiing_data_per_crop_test_set_create.py --g 1 --h 1  --chosen_crop_types 23
python traiing_data_per_crop_test_set_create.py --g 1 --h 1  --chosen_crop_types 28




Summer crops

	• GRAIN_MAIZE(8)
    • SUNFLOWER(30)
    • GRASSLAND(9)

    • SILAGE_MAIZE(18)
    • SOY(19)
    • FOREST(7)

export CUDA_VISIBLE_DEVICES=6
cd stelar_3dunet/
conda activate dt2
python traiing_data_per_crop_test_set_create.py --g 0 --h 0  --chosen_crop_types 8
python traiing_data_per_crop_test_set_create.py --g 0 --h 0  --chosen_crop_types 30
python traiing_data_per_crop_test_set_create.py --g 0 --h 0  --chosen_crop_types 9

python traiing_data_per_crop_test_set_create.py --g 0 --h 0  --chosen_crop_types 18
python traiing_data_per_crop_test_set_create.py --g 0 --h 0  --chosen_crop_types 19
python traiing_data_per_crop_test_set_create.py --g 0 --h 0  --chosen_crop_types 7



'''
import torch



# Image dimensions
rows, cols = 10002, 10002

# Define corner coordinates
ul_x, ul_y = 704855.0, 4995145.0  # upper left
ur_x, ur_y = 804875.0, 4995145.0  # upper right
ll_x, ll_y = 704855.0, 4895125.0  # lower left
lr_x, lr_y = 804875.0, 4895125.0  # lower right

x_coords = np.linspace(ul_x, ur_x, cols)  # same for top and bottom
y_coords = np.linspace(ul_y, ll_y, rows)  # same for left and right

x_grid, y_grid = np.meshgrid(x_coords, y_coords)

geospatial_image = np.stack((x_grid, y_grid), axis=-1)  # shape: (10002, 10002, 2)

print("Shape of geospatial image:", geospatial_image.shape)
print("Sample (0,0):", geospatial_image[0,0])  # Should be (704855.0, 4995145.0)
print("Sample (-1,-1):", geospatial_image[-1,-1])  # Should be (804875.0, 4895125.0)


device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
# start from potato
vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}



import argparse

parser = argparse.ArgumentParser(description='Crop speicific LAI and labels sampling')
parser.add_argument('--chosen_crop_types', type=int, default=3, help='Select crop type')
#parser.add_argument('--randcon', type=int, default=3, help='Select rancon')
parser.add_argument('--g', type=int, default=0, help='Select g')
parser.add_argument('--h', type=int, default=0, help='Select h')

args = parser.parse_args()


chosen_crop_types = args.chosen_crop_types
g = args.g
h = args.h
#randcon = args.randcon

logging.info(f"Selected crop type: {vista_crop_dict[chosen_crop_types]}")

#chosen_crop_types = 1

print("vista_crop_dict[chosen_crop_types]", vista_crop_dict[chosen_crop_types])


#labels_p = np.load('/home/luser/STELAR_Workbenches/saved_labels/lai_specific_france_mask_10_top.npy')
#labels = np.load('/home/luser/UniBw-STELAR/dataset/test_saves/vista_labes_image.npy').astype(np.uint8)
labels = np.load('./storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)
print("labels.shape", labels.shape)
#draw_time_index = random.randint(0, 245)
#selcted_time_files = filepaths[draw_time_index:draw_time_index+20]
#big_stack = torch.tensor([]).to(device)

#for k in range(100):

    #x_ind = random.choices(range(14), k=64)
    #y_ind = random.choices(range(14), k=64)


#time_strip_inds = [0, 52, 84, 116, 158, 190]
#time_strip_inds = [90, 100, 170, 178, 185, 190]
#time_strip_inds = [0, 40, 80, 120, 150, 175]  # for winter 
time_strip_inds = [20, 30, 120, 130, 175, 180] # for spring


def get_coord(i, j):
    x = ul_x + (ur_x - ul_x) * (j / (cols - 1))
    y = ul_y + (ll_y - ul_y) * (i / (rows - 1))
    return x, y

all_lai = torch.tensor([]).to(device)
all_labels = torch.tensor([]).to(device)
all_coords = torch.tensor([]).to(device)


for time_strip_ind in time_strip_inds:

    temporal_batches = torch.tensor([]).to(device)
    spatial_label_batches = torch.tensor([]).to(device)
    spatial_coord_batches = torch.tensor([]).to(device)


    print("len(filepaths)", len(filepaths))

    #time_strip_ind =  np.random.randint(0, len(filepaths)-65)

    print("time_strip_ind", time_strip_ind)

    considered_filepaths = filepaths[time_strip_ind:time_strip_ind+64]

    #for g in range(1):
    #for h in range(1):
    temporal_strip = torch.tensor([]).to(device)

    step = 0
    for filepath in considered_filepaths:
        numpy_array = np.load(filepath)
        torch_tensor = torch.from_numpy(numpy_array).to(device)
        step+=1
        numpy_array = 0.0
        temporal_strip = torch.cat((temporal_strip, torch_tensor[5000*g:5000*g+5000, 5000*h:5000*h+5000].unsqueeze(0)), axis=0)

    labels_ = labels[5000*g:5000*g+5000, 5000*h:5000*h+5000]
    geospatial_image = geospatial_image[5000*g:5000*g+5000, 5000*h:5000*h+5000]

    x_inds, y_inds = np.where(labels_==chosen_crop_types)

    print("(len(x_inds)!=0 and len(y_inds)!=0)", (len(x_inds)!=0 and len(y_inds)!=0))

    if(len(x_inds)!=0 and len(y_inds)!=0):

        print("len(x_inds)", len(x_inds))
        print("len(y_inds)", len(y_inds))

        for i in range(50):

            random_corner = random.choices(range(len(x_inds)-2), k=1)[0]
            x_corner = x_inds[random_corner]
            y_corner = y_inds[random_corner]

            if(x_corner>labels_.shape[0]-90):
                x_corner = x_corner-90
            if(y_corner>labels_.shape[1]-90):
                y_corner = y_corner-90

            space_label = labels_[x_corner:x_corner+64, y_corner:y_corner+64]
            space_coord = geospatial_image[x_corner:x_corner+64, y_corner:y_corner+64]

            space_label = torch.from_numpy(space_label).to(device)
            space_coord = torch.from_numpy(space_coord).to(device)

            print("space_coord.shape", space_coord.shape)

            print(space_coord[0,0,:])
            print(space_coord[0,-1,:])
            print(space_coord[-1,0,:])
            print(space_coord[-1,-1,:])

            space_coord = torch.stack([
                space_coord[0, 0, :],
                space_coord[0, -1, :],
                space_coord[-1, 0, :],
                space_coord[-1, -1, :]
            ], dim=0)

            print("coord_corners.shape", space_coord.shape)
            print("coord_corners", space_coord)
            temporal_strip_ = temporal_strip[:, x_corner:x_corner+64, y_corner:y_corner+64]


            spatial_label_batches = torch.cat((spatial_label_batches, space_label.unsqueeze(0)), axis=0)

            spatial_coord_batches = torch.cat((spatial_coord_batches, space_coord.unsqueeze(0)), axis=0)


            temporal_batches = torch.cat((temporal_batches, temporal_strip_.unsqueeze(0)), axis=0)

    else:    
        temporal_strip = torch.tensor([]).to(device)

    print("spatial_label_batches.shape", spatial_label_batches.shape)   
    print("temporal_batches.shape", temporal_batches.shape)    

    all_lai = torch.cat((all_lai, temporal_batches), axis=0)
    all_labels = torch.cat((all_labels, spatial_label_batches), axis=0) 
    all_coords = torch.cat((all_coords, spatial_coord_batches), axis=0) 


    print("all_lai.shape", all_lai.shape)   
    print("all_labels.shape", all_labels.shape)    
    print("all_coords.shape", all_coords.shape)    



array_lai = all_lai.cpu().numpy()

print("Last all_lai.shape", all_lai.shape)   
print("Last all_labels.shape", all_labels.shape)    
print("Last all_coords.shape", all_coords.shape)    

tifffile.imsave('./storage/per_crop_data_labels_test_g_'+str(g)+'_h_'+str(h)+'/'+vista_crop_dict[chosen_crop_types]+'/train'+vista_crop_dict[chosen_crop_types]+'n.tif', array_lai)


array_lab = all_labels.cpu().numpy()
tifffile.imsave('./storage/per_crop_data_labels_test_g_'+str(g)+'_h_'+str(h)+'/'+vista_crop_dict[chosen_crop_types]+'/lab'+vista_crop_dict[chosen_crop_types]+'n.tif', array_lab)


array_coord = all_coords.cpu().numpy()
tifffile.imsave('./storage/per_crop_data_labels_test_g_'+str(g)+'_h_'+str(h)+'/'+vista_crop_dict[chosen_crop_types]+'/coord'+vista_crop_dict[chosen_crop_types]+'n.tif', array_coord)


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


