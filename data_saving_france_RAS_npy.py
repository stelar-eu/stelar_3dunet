
'''
Required installations for the code to run:
conda install conda-forge::opencv


conda install conda-forge::opencv


'''


'''

export CUDA_VISIBLE_DEVICES=1
cd stelar_3dunet
conda activate stcon4
python data_saving_france_RAS_npy.py

'''



from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file, extract_spec_LAI_from_RAS_file, get_cluster_length
import matplotlib.pyplot as plt
#import torch
import numpy as np
datapath = './dataset/france2/lai_ras/'


image_length = 10002

image_width = 10002

select_image = 0

#for i in range(1):
#k=i+1
year = 20
month=1


import glob
filepaths = glob.glob('./dataset/france2/lai_ras/*.RAS')

filepaths.sort()

print("filepaths", filepaths)

datapath_filename = filepaths[0]
#filename = '32UQV_2002.RAS'
print("datapath_filename", datapath_filename)
for datapath_filename in filepaths:
    print("datapath_filename", datapath_filename)
    cluster_len = get_cluster_length(datapath_filename, image_length, image_width)
    for cluster_ind in range(cluster_len):
        test = extract_spec_LAI_from_RAS_file(datapath_filename, cluster_ind, image_length, image_width)
        print("test.shape", test.shape)
        
        test[test<0] = 0  
        #for i in range(len(test)):
        #print("test.max(), test.min()", test.max(), test.min())
        print("test no : "+str(cluster_ind)+" : ", test.shape)
        #save numpy array as .npy file
        print("datapath_filename[-14:-4]+'_measure_'+str(i) : ", datapath_filename[-14:-4]+'_measure_'+str(cluster_ind).zfill(2))
        np.save('./dataset/france2/processed_lai_npy/'+datapath_filename[-14:-4]+'_measure_'+str(cluster_ind).zfill(2)+'.npy', test)


print("len(test)", len(test))
print('test[0].shape', test[0].shape)
print('test[1].shape', test[1].shape)

test = test[0]

test[test<0] = 0

# convert test to numpy array
test_distri = np.array(test)

print('test_distri.max()', test_distri.max())
print('test_distri.max()', test_distri.min())
# plot the histogram of the values in the numpy array test
plt.hist(test_distri.flatten(), bins=100)
plt.savefig('./view_check/distri.png')
plt.close()



print('test.shape', test.shape)
plt.imshow(test)
plt.colorbar()
plt.savefig('./view_check/test1.png')
plt.close()




'''filename = '32UQV_2001.RAS'
test = extract_LAI_from_RAS_file(datapath, filename, image_length, image_width, select_image)
test[test<0] = 0

print('test.shape', test.shape)
plt.imshow(test)
plt.colorbar()
plt.savefig('/media/chethan/New Volume/1 A FI CODE/stelar/view_check/test'+str(k)+'.png')
plt.close()'''

#filepath = '/home/luser/Stelar project/dataset/lai_ras/32UQV_2001.RAS'

#explore_image(filepath)