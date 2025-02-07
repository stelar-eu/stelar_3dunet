import numpy as np


test_lai = np.load('/mdadm0/chethan_krishnamurth/stelar_3dunet/dataset/france2/processed_lai_npy/Q_LAI_2020_measure_00.npy')



print("test_lai.shape", test_lai.shape)


unique_elements, element_counts = np.unique(test_lai, return_counts=True)


print("unique_elements", unique_elements)

print("len(unique_elements)", len(unique_elements))

print("element_counts", element_counts)