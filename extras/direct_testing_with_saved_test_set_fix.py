import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
from keras.models import load_model
from scipy.stats import mode


import tensorflow as tf
device_name = tf.test.gpu_device_name()


import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


import segmentation_models_3D as sm
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tifffile

'''


export CUDA_VISIBLE_DEVICES=5
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python3 direct_testing_with_saved_test_set_fix.py --g 0 --h 0 --season winter
python3 direct_testing_with_saved_test_set_fix.py --g 1 --h 0 --season winter
python3 direct_testing_with_saved_test_set_fix.py --g 0 --h 1 --season winter
python3 direct_testing_with_saved_test_set_fix.py --g 1 --h 1 --season winter


python3 direct_testing_with_saved_test_set_fix.py --g 1 --h 1 --season spring

'''
'''

mixture of experts existing resuof already trained models: 


--crop_1 33 --crop_2 36 --crop_3 41
--crop_1 34 --crop_2 37 --crop_3 40


    chosen_crop_types_list_list = [[33, 34, 36], [37, 40, 41]]
    chosen_crop_types_list_list_models = [[33, 34, 36], [37, 40, 41]]
        #num_epochs = 150


        
spring types:

chosen_crop_types_list_list = [[2, 15, 20], [21, 23, 28]]


'''


import argparse
parser = argparse.ArgumentParser(description='Enter crop type numbers in order')

parser.add_argument('--g', type=int, default=0, help='Select crop type')
parser.add_argument('--h', type=int, default=0, help='Select crop type')
parser.add_argument('--season', type=str, default=0, help='input season')

args = parser.parse_args()

g = args.g
h = args.h
season = args.season

class_weights =  True
cloud_interpolation = False

vista_crop_dict = {0:'NA', 10:'NA' , 11: 'ALFALFA', 12: 'BEET', 13: 'CLOVER', 14: 'FLAX', 15: 'FLOWERING_LEGUMES', 16: 'FLOWERS', 17: 'FOREST', 18: 'GRAIN_MAIZE', 19: 'GRASSLAND', 20: 'HOPS', 21: 'LEGUMES', 22: 'VISTA_NA', 23: 'PERMANENT_PLANTATIONS', 24: 'PLASTIC', 25: 'POTATO', 26: 'PUMPKIN', 27: 'RICE', 28: 'SILAGE_MAIZE', 29: 'SOY', 30: 'SPRING_BARLEY', 31: 'SPRING_OAT', 32: 'SPRING_OTHER_CEREALS', 33: 'SPRING_RAPESEED', 34: 'SPRING_RYE', 35: 'SPRING_SORGHUM', 36: 'SPRING_SPELT', 37: 'SPRING_TRITICALE', 38: 'SPRING_WHEAT', 39: 'SUGARBEET', 40: 'SUNFLOWER', 41: 'SWEET_POTATOES', 42: 'TEMPORARY_GRASSLAND', 43: 'WINTER_BARLEY', 44: 'WINTER_OAT', 45: 'WINTER_OTHER_CEREALS', 46: 'WINTER_RAPESEED', 47: 'WINTER_RYE', 48: 'WINTER_SORGHUM', 49: 'WINTER_SPELT', 50: 'WINTER_TRITICALE', 51: 'WINTER_WHEAT'}


#season = "winter"
#season = "spring"

winter_crop_types = [10, 43, 44, 46, 47, 50, 51]
spring_crop_types = [10, 12, 25, 30, 31, 33, 38]


if season == "spring":
    chosen_season_crops = spring_crop_types
if season == "winter":
    chosen_season_crops = winter_crop_types


#g = 0
#h = 0
#winter_crop_types = [0, 33, 34, 35, 36, 37, 40, 41]


def get_labels_in_color(groud_truth_image):
    color_map = {10: [0, 0, 0], 0: [0, 0, 0], 11: [0, 255, 0], 12: [0, 0, 255], 13: [255, 255, 0], 14: [255, 165, 0], 15: [255, 0, 255], 16: [0, 255, 255], 17: [128, 0, 128], 18: [128, 128, 0], 19: [0, 128, 0], 20: [128, 0, 0], 21: [0, 0, 128], 22: [128, 128, 128], 23: [0, 128, 128], 24: [255, 0, 0], 25: [255, 255, 255], 26: [192, 192, 192], 27: [139, 0, 0], 28: [0, 100, 0], 29: [0, 0, 139], 30: [255, 215, 0], 31: [255, 140, 0], 32: [139, 0, 139], 33: [0, 206, 209], 34: [75, 0, 130], 35: [85, 107, 47], 36: [34, 139, 34], 37: [165, 42, 42], 38: [70, 130, 180], 39: [169, 169, 169], 40: [32, 178, 170], 41: [47, 79, 79], 42: [245, 245, 245], 43: [105, 105, 105], 44: [205, 92, 92], 45: [50, 205, 50], 46: [65, 105, 225], 47: [255, 223, 0], 48: [255, 99, 71], 49: [186, 85, 211], 50: [0, 191, 255], 51: [192, 192, 192]}

    groud_truth_color_image = np.zeros(groud_truth_image.shape + (3,), dtype=np.uint8)
    for i in range(groud_truth_image.shape[0]):
        for j in range(groud_truth_image.shape[1]):
            segment_id_gt = groud_truth_image[i, j]
            groud_truth_color_image[i, j] = color_map[segment_id_gt]
    return groud_truth_color_image


def replace_zeros_with_average(arr):
    for i in range(len(arr)):
        if arr[i] == 0:
            left = i - 1
            right = i + 1
            while left >= 0 and arr[left] == 0:
                left -= 1
            while right < len(arr) and arr[right] == 0:
                right += 1
            if left >= 0 and right < len(arr):
                arr[i] = (arr[left] + arr[right]) // 2
            elif left >= 0:
                arr[i] = arr[left]
            elif right < len(arr):
                arr[i] = arr[right]
    return arr

color_map = {10: [0, 0, 0], 0: [0, 0, 0], 11: [0, 255, 0], 12: [0, 0, 255], 13: [255, 255, 0], 14: [255, 165, 0], 15: [255, 0, 255], 16: [0, 255, 255], 17: [128, 0, 128], 18: [128, 128, 0], 19: [0, 128, 0], 20: [128, 0, 0], 21: [0, 0, 128], 22: [128, 128, 128], 23: [0, 128, 128], 24: [255, 0, 0], 25: [255, 255, 255], 26: [192, 192, 192], 27: [139, 0, 0], 28: [0, 100, 0], 29: [0, 0, 139], 30: [255, 215, 0], 31: [255, 140, 0], 32: [139, 0, 139], 33: [0, 206, 209], 34: [75, 0, 130], 35: [85, 107, 47], 36: [34, 139, 34], 37: [165, 42, 42], 38: [70, 130, 180], 39: [169, 169, 169], 40: [32, 178, 170], 41: [47, 79, 79], 42: [245, 245, 245], 43: [105, 105, 105], 44: [205, 92, 92], 45: [50, 205, 50], 46: [65, 105, 225], 47: [255, 223, 0], 48: [255, 99, 71], 49: [186, 85, 211], 50: [0, 191, 255], 51: [192, 192, 192]}


import gc
BACKBONE = 'vgg16'  
preprocess_input = sm.get_preprocessing(BACKBONE)




if class_weights:

    if(season=="winter"):
        chosen_crop_types_list_list = [[43, 46, 51], [44, 47, 50]] # winter crop inds
        chosen_crop_types_list_list_models = [[43, 46, 51], [44, 47, 50]]

    if(season=="spring"):
        chosen_crop_types_list_list = [[12, 25, 30], [31, 33, 38]] # spring crop inds
        chosen_crop_types_list_list_models = [[12, 25, 30], [31, 33, 38]]

else:

    if(season=="winter"):
        chosen_crop_types_list_list = [[43, 46, 51], [44, 47, 50]] # winter crop inds
        chosen_crop_types_list_list_models = [[43, 46, 51], [44, 47, 50]]

    if(season=="spring"):
        chosen_crop_types_list_list = [[12, 25, 30], [31, 33, 38]] # spring crop inds
        chosen_crop_types_list_list_models = [[12, 25, 30], [31, 33, 38]]



crop_types_all_list = [ 10, 11,  12,  13,  14,  15,  17,  18,  19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 37, 38, 40, 42, 43, 44, 45, 46, 47, 50, 51]

#chosen_crop_types_list = winter_crop_types




test_img_number = 159
chosen_subset_for_test_set = 4

for test_img_number in range(90):
    for chosen_subset_for_test_set in range(len(chosen_crop_types_list_list)):

        stacked_test_preds = []
        num_epochs = 850

        X_test = io.imread('./storage/test_sets_of_subsets_all_g_'+str(g)+'_h_'+str(h)+'/X_test_contains'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][0]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][1]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][2]]+'_.tif')
        y_test_1 = io.imread('./storage/test_sets_of_subsets_all_g_'+str(g)+'_h_'+str(h)+'/y_test_all_contains'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][0]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][1]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][2]]+'_.tif')

        print("y_test_1.shape", y_test_1.shape)

        ground_truth_1 = y_test_1[test_img_number-1] + 10

        for k in crop_types_all_list:
            if not(k in chosen_season_crops):
                ground_truth_1[ground_truth_1==k]=0


        ensambled_ground_truth = np.zeros((64, 64))
        ensambled_result_image = np.zeros((64, 64))
        ensambled_result_image_gen = np.zeros((64, 64, 64))

        test_img = X_test[test_img_number-1]
        tifffile.imsave('./ensamble_results/iou_f1_class_weights/inputs/input_subset_g'+season+''+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', test_img)

        if cloud_interpolation:
            for i in range(64):
                for j in range(64):
                    for c in range(3):
                        exp_time_strip = test_img[:, i, j, c]
                        exp_time_strip_av = replace_zeros_with_average(exp_time_strip)
                        test_img[:, i, j, c] = exp_time_strip_av
        test_img = test_img.astype(np.uint8)
        test_img_input=np.expand_dims(test_img, 0)
        test_img_input1 = preprocess_input(test_img_input, backend='tf')

        ground_truth_flattened = ground_truth_1[0,:,:]

        for chosen_crop_types_list in chosen_crop_types_list_list_models:

            gc.collect()
            if class_weights:
                my_model_1 = load_model('./storage/data_gen_model/final/3D_unet_g_'+str(g)+'_h_'+str(h)+'_labels_'+vista_crop_dict[chosen_crop_types_list[0]]+'_'+vista_crop_dict[chosen_crop_types_list[1]]+'_'+vista_crop_dict[chosen_crop_types_list[2]]+'_epoch_'+str(num_epochs)+'.h5', compile=False)
            else:
                my_model_1 = load_model('./storage/saved_model/3D_unet_g_'+str(g)+'_h_'+str(h)+'_labels_'+vista_crop_dict[chosen_crop_types_list[0]]+'_'+vista_crop_dict[chosen_crop_types_list[1]]+'_'+vista_crop_dict[chosen_crop_types_list[2]]+'_epoch_'+str(num_epochs)+'.h5', compile=False)

            test_pred1 = my_model_1.predict(test_img_input1)
            del my_model_1
            gc.collect()
            test_prediction1 = np.argmax(test_pred1, axis=4)[0,:,:,:]

            for i in range(3):
                test_prediction1[test_prediction1==i+1]= chosen_crop_types_list[i]
            for i in range(3):
                ensambled_result_image_gen[test_prediction1==chosen_crop_types_list[i]]=chosen_crop_types_list[i]
            stacked_test_preds.append(ensambled_result_image_gen.copy())
            ensambled_result_image_gen = np.zeros((64, 64, 64))

        stacked_test_preds = np.concatenate((stacked_test_preds), axis=0)
        ensambled_result_image_gen = np.median(stacked_test_preds, axis=0).astype(np.uint8)

        test_mode = np.zeros((64, 64))
        for i in range(64):
            for j in range(64):
                see = stacked_test_preds[:, i, j]
                sum_see = np.sum(see)
                if sum_see == 0:
                    test_mode[i, j] = 0
                else:
                    local_mode = mode(see[see>0])[0]
                    test_mode[i, j]= local_mode
                    
        test_mode = test_mode.astype(np.uint8)
        ground_truth_flattened = ground_truth_flattened.astype(np.uint8)

        # saving ground truth and prediction to calculate IOU and F1 score later        
        if cloud_interpolation:
            tifffile.imsave('./ensamble_results/iou_f1_class_weights/cloud_interpol/ground_truth_subset_g'+season+''+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', test_mode)
            tifffile.imsave('./ensamble_results/iou_f1_class_weights/cloud_interpol/prediction_subset_g'+season+''+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', ground_truth_flattened)
        else:
            tifffile.imsave('./ensamble_results/iou_f1_class_weights/no_cloud_interpol/ground_truth_subset_g'+season+''+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', test_mode)
            tifffile.imsave('./ensamble_results/iou_f1_class_weights/no_cloud_interpol/prediction_subset_g'+season+''+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', ground_truth_flattened)
        test_mode_c = get_labels_in_color(test_mode)

        '''for k in crop_types_all_list:
            if not(k in chosen_crop_types_list):
                ground_truth_flattened[ground_truth_flattened==k]=0'''
        
        print("5: np.unique(ground_truth_flattened)", np.unique(ground_truth_flattened))
        ensambled_ground_truth_c = get_labels_in_color(ground_truth_flattened)


        plt.figure(figsize=(10,10))
        plt.title(' Time series LAI input')
        for i in range(64):
            plt.axis('off')
            plt.subplot(8, 8, i+1)
            plt.imshow(test_img[i,:,:, 1])
            plt.axis('off')
        if cloud_interpolation:
            plt.savefig('./ensamble_results/after_class_weights_with_interpolation/'+season+'LAI_subset_g'+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.png', bbox_inches='tight')
        else:
            plt.savefig('./ensamble_results/after_class_weights_without_interpolation/'+season+'LAI_subset_g'+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.png', bbox_inches='tight')
        #plt.show()
        plt.close()


        plt.figure(figsize=(12, 8))
        plt.subplot(232)
        plt.title(' Ground truth')
        plt.imshow(ensambled_ground_truth_c)
        plt.axis('off')

        legend_elements = []
        concatenated_ensambles = np.concatenate((ground_truth_flattened, test_mode))
        uniq_crop_vals = np.unique(concatenated_ensambles)
        print("uniq_crop_vals", uniq_crop_vals)
        for k in uniq_crop_vals:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=[value / 255 for value in color_map[k]] , markersize=10, label=vista_crop_dict[k]))
        plt.subplot(233)
        plt.title('Prediction')
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.imshow(test_mode_c)

        if cloud_interpolation:
            plt.savefig('./ensamble_results/after_class_weights_with_interpolation/'+season+'ground_truth_pred_subset_g'+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.png', bbox_inches='tight')
        else:
            plt.savefig('./ensamble_results/after_class_weights_without_interpolation/'+season+'ground_truth_pred_subset_g'+str(g)+'h'+str(h)+''+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.png', bbox_inches='tight')
        #plt.show()
        plt.close()


