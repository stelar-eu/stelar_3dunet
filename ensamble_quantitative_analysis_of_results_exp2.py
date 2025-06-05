import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
from keras.models import load_model
import glob
from scipy.stats import mode




'''

export CUDA_VISIBLE_DEVICES=6
conda deactivate
conda deactivate
conda activate stcon4
cd stelar_3dunet/
python ensamble_quantitative_analysis_of_results_exp2.py

'''


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, accuracy_score



import tifffile



#input_img0 = np.load('/home/luser/UniBw-STELAR/dataset/france2/processed_lai_npy/Q_LAI_2020_measure_00.npy')
#input_img1 = np.load('./dataset/france2/processed_lai_npy/Q_LAI_2020_measure_00.npy')



import numpy as np

#input_img0_float32 = input_img0.astype(np.float32)
#print(input_img0_float32.std())



## Vista Crop Types

'''
    chosen_crop_types_list_list = [[2, 15, 20], [21, 23, 28]]

    chosen_crop_types_list_list_models = [[34, 35, 37], [33, 36, 40], [33, 36, 41]]


'''

vista_crop_dict = {0:'NA', 10:'NA' , 11: 'ALFALFA', 12: 'BEET', 13: 'CLOVER', 14: 'FLAX', 15: 'FLOWERING_LEGUMES', 16: 'FLOWERS', 17: 'FOREST', 18: 'GRAIN_MAIZE', 19: 'GRASSLAND', 20: 'HOPS', 21: 'LEGUMES', 22: 'VISTA_NA', 23: 'PERMANENT_PLANTATIONS', 24: 'PLASTIC', 25: 'POTATO', 26: 'PUMPKIN', 27: 'RICE', 28: 'SILAGE_MAIZE', 29: 'SOY', 30: 'SPRING_BARLEY', 31: 'SPRING_OAT', 32: 'SPRING_OTHER_CEREALS', 33: 'SPRING_RAPESEED', 34: 'SPRING_RYE', 35: 'SPRING_SORGHUM', 36: 'SPRING_SPELT', 37: 'SPRING_TRITICALE', 38: 'SPRING_WHEAT', 39: 'SUGARBEET', 40: 'SUNFLOWER', 41: 'SWEET_POTATOES', 42: 'TEMPORARY_GRASSLAND', 43: 'WINTER_BARLEY', 44: 'WINTER_OAT', 45: 'WINTER_OTHER_CEREALS', 46: 'WINTER_RAPESEED', 47: 'WINTER_RYE', 48: 'WINTER_SORGHUM', 49: 'WINTER_SPELT', 50: 'WINTER_TRITICALE', 51: 'WINTER_WHEAT'}
#chosen_crop_types_list_list = [[1, 2, 3], [4, 5, 7], [8,  9, 10], [11, 12, 13], [14, 15, 16], [18, 19, 20], [21, 23, 27], [28, 30, 32], [33, 34, 35], [36, 37, 40], [37, 40, 41]]



# load all the files in ./ensamble_results_max/iou_f1_class_weights/cloud_interpol

cloud_interpol = False
'''if cloud_interpol:
    ground_truth_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights_mixture_experts/cloud_interpol/g*.tif')
    ground_truth_filepaths.sort()

    predictions_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights_mixture_experts/cloud_interpol/p*.tif')
    predictions_filepaths.sort()
else:
    ground_truth_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights_mixture_experts/no_cloud_interpol/g*.tif')
    ground_truth_filepaths.sort()

    predictions_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights_mixture_experts/no_cloud_interpol/p*.tif')
    predictions_filepaths.sort()'''


if cloud_interpol:
    ground_truth_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights/cloud_interpol/g*.tif')
    ground_truth_filepaths.sort()

    predictions_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights/cloud_interpol/p*.tif')
    predictions_filepaths.sort()
else:
    ground_truth_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights/no_cloud_interpol/g*.tif')
    ground_truth_filepaths.sort()

    predictions_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights/no_cloud_interpol/p*.tif')
    predictions_filepaths.sort()

    '''ground_truth_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights/no_cloud_interpol_b/g*.tif')
    ground_truth_filepaths.sort()

    predictions_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights/no_cloud_interpol_b/p*.tif')
    predictions_filepaths.sort()'''



#ground_truth_subset_seasonspring_0_sample_0_        prediction_subset_seasonspring_0_sample_0_
#ground_truth_subset_seasonsummer_autumn_0_sample_0_.       prediction_subset_seasonsummer_autumn_0_sample_0_
#ground_truth_subset_seasonwinter_0_sample_0_                  prediction_subset_seasonwinter_0_sample_0_
#ground_truth_subset_seasonwinter_spring_summer_0_sample_0_       prediction_subset_seasonwinter_spring_summer_0_sample_0_


all_test_ground_truth = []
all_test_prediction = []
for ground_truth_address, prediction_address in zip(ground_truth_filepaths, predictions_filepaths):
    ground_truth = io.imread(ground_truth_address)
    prediction = io.imread(prediction_address)
    all_test_ground_truth.append(ground_truth)
    all_test_prediction.append(prediction)  
all_test_ground_truth = np.array(all_test_ground_truth)
all_test_prediction = np.array(all_test_prediction)

print("all_test_ground_truth.shape, all_test_prediction.shape", all_test_ground_truth.shape, all_test_prediction.shape)




def calculate_iou(ground_truth, prediction):
    ground_truth = ground_truth.astype(np.bool_)
    prediction = prediction.astype(np.bool_)
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    '''if(np.sum(ground_truth)==0 and np.sum(prediction)==0):
        iou = 1.0'''
    if np.sum(union) == 0:
        iou = 0
    else:
        iou = np.sum(intersection) / np.sum(union)    
    return iou

def calculate_f1_score(ground_truth, prediction):
    ground_truth = ground_truth.astype(np.bool_)
    prediction = prediction.astype(np.bool_)
    tp = np.sum(np.logical_and(prediction, ground_truth))
    fp = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)))
    fn = np.sum(np.logical_and(np.logical_not(prediction), ground_truth))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_accuracy(ground_truth, prediction):
    ground_truth = ground_truth.astype(np.bool_)
    prediction = prediction.astype(np.bool_)
    
    tp = np.sum(np.logical_and(prediction, ground_truth))
    tn = np.sum(np.logical_and(np.logical_not(prediction), np.logical_not(ground_truth)))
    fp = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)))
    fn = np.sum(np.logical_and(np.logical_not(prediction), ground_truth))
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    return accuracy


## Quantitative Analysis : Crop prediction
### Sampled training data analysis



#crop_types_all_list = [ 1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 41]
#all_unique_crop_types = np.array([ 0,  1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 41]).astype(np.uint8)
#chosen_crop_types_list_list_models = [[33, 34, 36], [37, 40, 41]]
#all_unique_crop_types = np.array([ 33, 34, 36, 37, 40, 41]).astype(np.uint8)

#chosen_crop_types_list_list = [[33, 36, 41], [34, 37, 40]]

#all_unique_crop_types = np.array([ 33, 34, 36, 37, 40, 41]).astype(np.uint8)

all_unique_crop_types = np.array([12, 25, 30, 31, 33, 38, 43, 44, 46, 47, 50, 51, 18, 19, 40, 17, 28, 29, 14], dtype=np.uint8)
#all_unique_crop_types = np.array([12, 25, 30, 31, 33, 38, 43, 44, 46, 47, 50, 51], dtype=np.uint8)


# For permanent plantations no: 13 in the test set
all_crops_iou_distributions = []
#for unique in all_unique_crop_types[1:]:
for unique in all_unique_crop_types:
    each_iou_distribution = []
    for i in range(all_test_ground_truth.shape[0]):
        ground_truth, prediction = all_test_ground_truth[i].copy(), all_test_prediction[i].copy()
        ground_truth[ground_truth!=unique]=0
        prediction[prediction!=unique]=0

        ground_truth_b = ground_truth.astype(np.bool_)
        prediction_b = prediction.astype(np.bool_)
        #print("ground_truth_b.max()", ground_truth_b.max())
        #print("ground_truth_b.min()", ground_truth_b.min())
        print("np.sum(ground_truth_b)", np.sum(ground_truth_b))
        #if(np.sum(ground_truth_b)>400):
        if(not np.sum(ground_truth)==0 and not np.sum(prediction)==0):
        #if(not np.sum(ground_truth)==0):
            iou = calculate_iou(ground_truth, prediction)
            each_iou_distribution.append(iou)
    print("len(each_iou_distribution)", len(each_iou_distribution))
    all_crops_iou_distributions.append(each_iou_distribution)


# For permanent plantations no: 13 in the test set
all_crops_accuracy_distributions = []
#for unique in all_unique_crop_types[1:]:
for unique in all_unique_crop_types:
    each_acc_distribution = []
    for i in range(all_test_ground_truth.shape[0]):
        ground_truth, prediction = all_test_ground_truth[i].copy(), all_test_prediction[i].copy()
        ground_truth[ground_truth!=unique]=0
        prediction[prediction!=unique]=0

        ground_truth_b = ground_truth.astype(np.bool_)
        prediction_b = prediction.astype(np.bool_)
        #print("ground_truth_b.max()", ground_truth_b.max())
        #print("ground_truth_b.min()", ground_truth_b.min())
        print("np.sum(ground_truth_b)", np.sum(ground_truth_b))
        #if(np.sum(ground_truth_b)>400):
        #if(not np.sum(ground_truth)==0 and not np.sum(prediction)==0):
        if(not np.sum(ground_truth)==0):
            acc = calculate_accuracy(ground_truth, prediction)
            each_acc_distribution.append(acc)
    print("len(each_iou_distribution)", len(each_acc_distribution))
    all_crops_accuracy_distributions.append(each_acc_distribution)


# to get confusion matrix
for unique in all_unique_crop_types:
    each_iou_distribution = []
    for i in range(all_test_ground_truth.shape[0]):
        ground_truth, prediction = all_test_ground_truth[i].copy(), all_test_prediction[i].copy()
        print()
        print("ground_truth.shape", ground_truth.shape)
        print("prediction.shape", prediction.shape)
        print()
        print("ground_truth.max()", ground_truth.max())
        print("prediction.min()", prediction.min())
        print()
        print("uniques")
        print("np.unique(ground_truth)", np.unique(ground_truth))
        print("np.unique(prediction)", np.unique(prediction))




############# confusion matrix computation #############


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Unique crop types you're interested in
#all_unique_crop_types = np.array([2, 15, 20, 21, 23, 28, 33, 34, 36, 37, 40, 41], dtype=np.uint8)

#all_unique_crop_types = np.array([12, 25, 30, 31, 33, 38, 43, 44, 46, 47, 50, 51], dtype=np.uint8)
all_unique_crop_types = np.array([12, 25, 30, 31, 33, 38, 43, 44, 46, 47, 50, 51, 18, 19, 40, 17, 28, 29, 14], dtype=np.uint8)
#all_unique_crop_types = np.array([12, 25, 30, 31, 33, 38, 43, 44, 46, 47, 50, 51], dtype=np.uint8)


# Initialize empty lists to collect all flattened ground truth and prediction labels
all_gt_labels = []
all_pred_labels = []

# Loop through all test patches
for i in range(all_test_ground_truth.shape[0]):
    gt = all_test_ground_truth[i].copy().flatten()
    pred = all_test_prediction[i].copy().flatten()

    # Optional: Only keep labels that are in the desired crop list (and ignore background, e.g. 0)
    mask = np.isin(gt, all_unique_crop_types)  # You could also apply this to pred if needed
    gt = gt[mask]
    pred = pred[mask]

    all_gt_labels.append(gt)
    all_pred_labels.append(pred)

# Concatenate all flattened arrays into a single long vector
all_gt_labels = np.concatenate(all_gt_labels)
all_pred_labels = np.concatenate(all_pred_labels)

crop_labels = [vista_crop_dict[crop_id] for crop_id in all_unique_crop_types]
# Compute confusion matrix
#cm = confusion_matrix(all_gt_labels, all_pred_labels, labels=all_unique_crop_types)
cm = confusion_matrix(all_gt_labels, all_pred_labels, labels=all_unique_crop_types)

# Plot the confusion matrix
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_unique_crop_types)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=crop_labels)

fig, ax = plt.subplots(figsize=(15, 15))
disp.plot(ax=ax, cmap='viridis', xticks_rotation=45)
plt.title("Confusion Matrix of Crop Type Segmentation")
plt.savefig("showcase/confusion_matrix/crop_type_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()


############# confusion matrix computation #############




import matplotlib.pyplot as plt

# Example data
data = all_crops_iou_distributions
crop_types = [vista_crop_dict[element] for element in all_unique_crop_types]

# Create a new figure with a larger size
plt.figure(figsize=(14, 8))  # Increase figure size to make more room for x-ticks

# Create box plot with broader boxes
box = plt.boxplot(data, patch_artist=True, widths=0.6)  # Adjust widths to make boxes broader

# Customize box colors
#colors = plt.cm.Paired(range(len(data)))  # Using a colormap for distinct colors

# Generate distinct colors
cmap = plt.cm.get_cmap('tab20', len(data))  # tab20 has 20 distinct colors
colors = [cmap(i) for i in range(len(data))]


for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Customize the rest of the plot for better visibility
plt.title('IOU Distributions', fontsize=16)
plt.xlabel('Crop Types', fontsize=14)
plt.ylabel('Intersection over Union', fontsize=14)

# Set custom x-tick positions to spread them out
positions = range(1, len(data) + 1)
#plt.xticks(positions, crop_types[1:], rotation=90, fontsize=12)
plt.xticks(positions, crop_types, rotation=90, fontsize=12)

# Add gridlines for better readability
plt.grid(True, linestyle='--', linewidth=0.5)
if cloud_interpol:
    plt.savefig('./showcase_max/iou/exp2_iou_cloud_interpol.png', bbox_inches='tight')
else:
    plt.savefig('./showcase_max/iou/exp2_iou_no_cloud_interpol.png', bbox_inches='tight')

# Display the plot
plt.show()



## F1 Score


# For permanent plantations no: 13 in the test set
all_crops_f1_distributions = []
for unique in all_unique_crop_types:
    each_f1_distribution = []
    for i in range(all_test_ground_truth.shape[0]):
        ground_truth, prediction = all_test_ground_truth[i].copy(), all_test_prediction[i].copy()
        
        ground_truth[ground_truth!=unique]=0
        prediction[prediction!=unique]=0

        ground_truth_b = ground_truth.astype(np.bool_)
        prediction_b = prediction.astype(np.bool_)
        if(not np.sum(ground_truth)==0 and not np.sum(prediction)==0):
        #if(not np.sum(ground_truth)==0):
        #if(np.sum(ground_truth_b)>400):
            f1 = calculate_f1_score(ground_truth, prediction)
            each_f1_distribution.append(f1)
        #print("Intersection over union", iou)
    all_crops_f1_distributions.append(each_f1_distribution)



import matplotlib.pyplot as plt

# Example data
data = all_crops_f1_distributions
crop_types = [vista_crop_dict[element] for element in all_unique_crop_types]

# Create a new figure with a larger size
plt.figure(figsize=(14, 8))  # Increase figure size to make more room for x-ticks

# Create box plot with broader boxes
box = plt.boxplot(data, patch_artist=True, widths=0.6)  # Adjust widths to make boxes broader

# Customize box colors
#colors = plt.cm.Paired(range(len(data)))  # Using a colormap for distinct colors

# Generate distinct colors
cmap = plt.cm.get_cmap('tab20', len(data))  # tab20 has 20 distinct colors
colors = [cmap(i) for i in range(len(data))]

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Customize the rest of the plot for better visibility
plt.title('F1 Scores', fontsize=16)
plt.xlabel('Crop Types', fontsize=14)
plt.ylabel('F1 Scores', fontsize=14)

# Set custom x-tick positions to spread them out
positions = range(1, len(data) + 1)
#plt.xticks(positions, crop_types[1:], rotation=90, fontsize=12)
plt.xticks(positions, crop_types, rotation=90, fontsize=12)

# Add gridlines for better readability
plt.grid(True, linestyle='--', linewidth=0.5)
if cloud_interpol:
    plt.savefig('./showcase_max/f1_score/exp2_f1_cloud_interpol.png', bbox_inches='tight')
else:
    plt.savefig('./showcase_max/f1_score/exp2_f1_no_cloud_interpol.png', bbox_inches='tight')


# Display the plot
plt.show()
plt.close()




# Example data
data = all_crops_accuracy_distributions
crop_types = [vista_crop_dict[element] for element in all_unique_crop_types]

# Create a new figure with a larger size
plt.figure(figsize=(14, 8))  # Increase figure size to make more room for x-ticks

# Create box plot with broader boxes
box = plt.boxplot(data, patch_artist=True, widths=0.6)  # Adjust widths to make boxes broader

# Customize box colors
#colors = plt.cm.Paired(range(len(data)))  # Using a colormap for distinct colors

# Generate distinct colors
cmap = plt.cm.get_cmap('tab20', len(data))  # tab20 has 20 distinct colors
colors = [cmap(i) for i in range(len(data))]


for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Customize the rest of the plot for better visibility
plt.title('Accuarcy', fontsize=16)
plt.xlabel('Crop Types', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

# Set custom x-tick positions to spread them out
positions = range(1, len(data) + 1)
#plt.xticks(positions, crop_types[1:], rotation=90, fontsize=12)
plt.xticks(positions, crop_types, rotation=90, fontsize=12)

# Add gridlines for better readability
plt.grid(True, linestyle='--', linewidth=0.5)
if cloud_interpol:
    plt.savefig('./showcase_max/accuracy/exp2_acc_cloud_interpol.png', bbox_inches='tight')
else:
    plt.savefig('./showcase_max/accuracy/exp2_acc_no_cloud_interpol.png', bbox_inches='tight')


# Display the plot
plt.show()
plt.close()