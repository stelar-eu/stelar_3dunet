# stelar_3dunet
Bias mitigation strategies for agricultural crop type prediction using 3D U-net for spatio-temporal semantic segmentation



Code for 3D unet is adopted from Bhattiprolu, S. (2023). python_for_microscopists. GitHub. https://github.com/bnsreenu/python_for_microscopists/blob/master

Below is the dictionaly of crop types to predict: 

{ 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}


# Ensamble of models

## Experiment 0

This is a small scale example with lower coverage of landscape and less number of crop types. In this experiment we just considered the 10 percent of overall landscape. We consider corpp types ALFALFA, FOREST, GRAIN_MAIZE, GRASSLAND, LEGUMES, PERMANENT_PLANTATIONS and TEMPORARY_GRASSLAND and train three models to predict these crop types: 

Below are the results: 

### Considered crop type distribution

<img src="./showcase/small_scale_results/considered_distribution.png" alt="considered_distribution" width="600">


### Intersection over union for crop types

<img src="./showcase/small_scale_results/iou_exp0.png" alt="iou_exp0" width="600">


#### Qualitative results : 

1. Sample 1

<img src="./showcase/small_scale_results/LAI1.png" alt="LAI_1" width="400">
<img src="./showcase/small_scale_results/ground_pred1.png" alt="Ground Truth vs Prediction" width="600">


1. Sample 2

<img src="./showcase/small_scale_results/LAI2.png" alt="LAI_2" width="400">
<img src="./showcase/small_scale_results/ground_pred2.png" alt="Ground Truth vs Prediction" width="600">

1. Sample 3

<img src="./showcase/small_scale_results/LAI3.png" alt="LAI_2" width="400">
<img src="./showcase/small_scale_results/ground_pred3.png" alt="Ground Truth vs Prediction" width="600">


## Experiment 1 
We train 11 different models such that each model predicts three crop types. We then create an ensamble of models and post process a cosolidated crop type prediction by the ensamble of models on a given spatio-temporal patch. 

Each of the 11 models trained as follows :

model 1 : 
    types : NA, ALFALFA, BEET, CLOVER
    samples : 850 each

model 2 : 
    types : NA, FLAX, FLOWERING_LEGUMES, FOREST

    .
    .
    .
model 11:
    ...

#### Crop type representation in training data

<img src="./showcase/representation/crop_representatio_exp1.png" alt="crop_representatio_exp1" width="600">


### Intersection over union for crop types

<img src="./showcase/iou/exp1_iou.png" alt="exp1_iou" width="600">

#### Qualitative results : 

1. Sample 1

<img src="./showcase/ensamble_results/no_oversampling/LAI_1.png" alt="LAI_1" width="400">
<img src="./showcase/ensamble_results/no_oversampling/ground_truth_pred1.png" alt="Ground Truth vs Prediction" width="600">


2. Sample 2

<img src="./showcase/ensamble_results/no_oversampling/LAI_34.png" alt="LAI_34" width="400">
<img src="./showcase/ensamble_results/no_oversampling/ground_truth_pred34.png" alt="Ground Truth vs Prediction" width="600">


3. Sample 3

<img src="./showcase/ensamble_results/no_oversampling/LAI_46.png" alt="LAI_46" width="400">
<img src="./showcase/ensamble_results/no_oversampling/ground_truth_pred46.png" alt="Ground Truth vs Prediction" width="600">

4. Sample 4

<img src="./showcase/ensamble_results/no_oversampling/LAI_52.png" alt="LAI_52" width="400">
<img src="./showcase/ensamble_results/no_oversampling/ground_truth_pred52.png" alt="Ground Truth vs Prediction" width="600">


