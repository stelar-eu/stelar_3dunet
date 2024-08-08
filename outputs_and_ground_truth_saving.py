import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
from keras.models import load_model

from scipy.stats import mode
import gc
import tifffile
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

#Make sure the GPU is available. 
import tensorflow as tf
device_name = tf.test.gpu_device_name()


#Make sure the GPU is available. 
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



vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}

chosen_crop_types_list_list = [[1, 2, 3], [4, 5, 7], [8,  9, 10], [11, 12, 13], [14, 15, 16], [18, 19, 20], [21, 23, 27], [28, 30, 32], [33, 34, 35], [36, 37, 40], [37, 40, 41]]


num_epochs = 1500

all_images_iou = []
all_images_f1 = []


for chosen_crop_types_list_indata in chosen_crop_types_list_list:


    #chosen_crop_types_list = [1, 2, 3]

    print("vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[1]], vista_crop_dict[chosen_crop_types_list[1]]", vista_crop_dict[chosen_crop_types_list_indata[0]], vista_crop_dict[chosen_crop_types_list_indata[1]], vista_crop_dict[chosen_crop_types_list_indata[2]])

    all_input_img = []
    all_input_mask = []
    for i in chosen_crop_types_list_indata:
        input_img = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[i]+'/train_'+vista_crop_dict[i]+'.tif')
        input_mask = io.imread('/home/luser/stelar_3dunet/storage/per_crop_data_labels/'+vista_crop_dict[i]+'/lab_'+vista_crop_dict[i]+'.tif').astype(np.uint8)

        all_input_img.append(input_img)
        all_input_mask.append(input_mask)

    input_img = np.concatenate((all_input_img), axis=0).reshape(-1, 64, 64, 64)
    input_mask = np.concatenate((all_input_mask), axis=0).reshape(-1, 64, 64)

    input_mask = np.repeat(input_mask[:, np.newaxis, :, :], repeats=64, axis=1)


    print("input_img.shape", input_img.shape)
    print("input_mask.shape", input_mask.shape)



    unique_elements, element_counts = np.unique(input_mask, return_counts=True)


    #clear memory
    del all_input_img 
    del all_input_mask



    lai_uniques = np.unique(input_img)

    for n in range(len(lai_uniques)):
        input_img[input_img==lai_uniques[n]]=n
    input_img = input_img.astype(np.uint8)



    # clear memory 
    del lai_uniques


    n_classes=4

    BACKBONE = 'vgg16'  #Try vgg16, efficientnetb7, inceptionv3, resnet50
    #BACKBONE = 'resnet50'  #Try vgg16, efficientnetb7, inceptionv3, resnet50

    preprocess_input = sm.get_preprocessing(BACKBONE)


    def get_labels_in_color(groud_truth_image):
        color_map = {
            0: [0, 0, 0],1: [0, 255, 0], 2: [0, 0, 255], 3: [255, 255, 0], 4: [255, 165, 0], 5: [255, 0, 255], 6: [0, 255, 255],   
            7: [128, 0, 128], 8: [128, 128, 0], 9: [0, 128, 0], 10: [128, 0, 0], 11: [0, 0, 128], 12: [128, 128, 128], 13: [0, 128, 128],   
            14: [255, 0, 0], 15: [255, 255, 255], 16: [192, 192, 192], 17: [255, 0, 0], 18: [0, 255, 0], 19: [0, 0, 255], 20: [255, 255, 0],   
            21: [255, 165, 0], 22: [255, 0, 255],  23: [0, 255, 255],  24: [128, 0, 128],  25: [128, 128, 0],  26: [0, 128, 0],     
            27: [128, 0, 0],  28: [0, 0, 128], 29: [128, 128, 128], 30: [0, 128, 128], 31: [0, 0, 0], 32: [255, 255, 255], 
            33: [192, 192, 192], 34: [255, 0, 0], 35: [0, 255, 0], 36: [0, 0, 255], 37: [255, 255, 0], 38: [255, 165, 0], 
            39: [255, 0, 255],  40: [0, 128, 255],  41: [192, 192, 192] }
        groud_truth_color_image = np.zeros(groud_truth_image.shape + (3,), dtype=np.uint8)
        for i in range(groud_truth_image.shape[0]):
            for j in range(groud_truth_image.shape[1]):
                segment_id_gt = groud_truth_image[i, j]
                groud_truth_color_image[i, j] = color_map[segment_id_gt]
        return groud_truth_color_image

    color_map = {
        0: [0, 0, 0],1: [0, 255, 0], 2: [0, 0, 255], 3: [255, 255, 0], 4: [255, 165, 0], 5: [255, 0, 255], 6: [0, 255, 255],   
        7: [128, 0, 128], 8: [128, 128, 0], 9: [0, 128, 0], 10: [128, 0, 0], 11: [0, 0, 128], 12: [128, 128, 128], 13: [0, 128, 128],   
        14: [255, 0, 0], 15: [255, 255, 255], 16: [192, 192, 192], 17: [255, 0, 0], 18: [0, 255, 0], 19: [0, 0, 255], 20: [255, 255, 0],   
        21: [255, 165, 0], 22: [255, 0, 255],  23: [0, 255, 255],  24: [128, 0, 128],  25: [128, 128, 0],  26: [0, 128, 0],     
        27: [128, 0, 0],  28: [0, 0, 128], 29: [128, 128, 128], 30: [0, 128, 128], 31: [0, 0, 0], 32: [255, 255, 255], 
        33: [192, 192, 192], 34: [255, 0, 0], 35: [0, 255, 0], 36: [0, 0, 255], 37: [255, 255, 0], 38: [255, 165, 0], 
        39: [255, 0, 255],  40: [0, 128, 255],  41: [192, 192, 192] }



    for test_img_number in range(30):

        input_mask_1 = input_mask.copy()
        train_img = np.stack((input_img,)*3, axis=-1)
        ensambled_ground_truth = np.zeros((64, 64))
        ensambled_result_image = np.zeros((64, 64))
        ensambled_result_image_gen = np.zeros((64, 64, 64))

        stacked_test_preds = []
        nnn = 0
        for chosen_crop_types_list in chosen_crop_types_list_list:

            input_mask_1[input_mask_1<chosen_crop_types_list[0]]=0
            input_mask_1[input_mask_1>chosen_crop_types_list[-1]]=0
            for i in range(3):
                input_mask_1[input_mask_1==chosen_crop_types_list[i]]=i+1
            train_mask = np.expand_dims(input_mask_1, axis=4)
            train_mask_cat = to_categorical(train_mask, num_classes=n_classes)
            X_train, X_test, y_train, y_test_1 = train_test_split(train_img, train_mask_cat, test_size = 0.10, random_state = 0)
            print("y_test_1.shape", y_test_1.shape)
            del X_train
            del train_mask_cat
            del train_mask
            del y_train
            del input_mask_1
            gc.collect()
            my_model_1 = load_model('/home/luser/stelar_3dunet/storage/saved_model/3D_unet_res_labels_'+vista_crop_dict[chosen_crop_types_list[0]]+'_'+vista_crop_dict[chosen_crop_types_list[1]]+'_'+vista_crop_dict[chosen_crop_types_list[2]]+'_num_epocs_'+str(num_epochs)+'.h5', compile=False)

            test_img = X_test[test_img_number-1]
            ground_truth_1 = y_test_1[test_img_number-1]
            del y_test_1
            ground_truth_argmax_1 = np.argmax(ground_truth_1, axis=3)
            del ground_truth_1
            test_img_input=np.expand_dims(test_img, 0)
            test_img_input1 = preprocess_input(test_img_input, backend='tf')
            del test_img_input
            test_pred1 = my_model_1.predict(test_img_input1)
            del test_img_input1
            del my_model_1
            gc.collect()
            test_prediction1 = np.argmax(test_pred1, axis=4)
            test_prediction1 = np.argmax(test_pred1, axis=4)[0,:,:,:]
            
            for i in range(3):
                test_prediction1[test_prediction1==i+1]= chosen_crop_types_list[i]
                ground_truth_argmax_1[ground_truth_argmax_1==i+1]= chosen_crop_types_list[i]
            #result_image1 = np.median(test_prediction1, axis=0).astype(np.uint8)
            ground_truth_image1 = np.median(ground_truth_argmax_1, axis=0).astype(np.uint8)

            for i in range(3):
                ensambled_ground_truth[ground_truth_image1==chosen_crop_types_list[i]]=chosen_crop_types_list[i]
                #ensambled_result_image[result_image1==chosen_crop_types_list[i]]=chosen_crop_types_list[i]
                ensambled_result_image_gen[test_prediction1==chosen_crop_types_list[i]]=chosen_crop_types_list[i]
                
            stacked_test_preds.append(ensambled_result_image_gen.copy())
            ensambled_result_image_gen = np.zeros((64, 64, 64))

            input_mask_1 = input_mask.copy()
            nnn = nnn+1

        stacked_test_preds = np.concatenate((stacked_test_preds), axis=0)
        #ensambled_result_image_gen = np.median(stacked_test_preds, axis=0).astype(np.uint8)

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


        print("test_mode.shape", test_mode.shape)
        print("ensambled_ground_truth.shape", ensambled_ground_truth.shape)

        ground_truth = ensambled_ground_truth
        prediction = test_mode

        tifffile.imsave('/home/luser/stelar_3dunet/ensamble_results/iou_f1/ground_truth_'+str(test_img_number)+'_contains'+vista_crop_dict[chosen_crop_types_list_indata[0]]+'_'+vista_crop_dict[chosen_crop_types_list_indata[1]]+'_'+vista_crop_dict[chosen_crop_types_list_indata[2]]+'_.tif', ground_truth)
        tifffile.imsave('/home/luser/stelar_3dunet/ensamble_results/iou_f1/prediction_'+str(test_img_number)+'_contains'+vista_crop_dict[chosen_crop_types_list_indata[0]]+'_'+vista_crop_dict[chosen_crop_types_list_indata[1]]+'_'+vista_crop_dict[chosen_crop_types_list_indata[2]]+'_.tif', prediction)