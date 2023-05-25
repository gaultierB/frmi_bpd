import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

example_filename = "sub-EESS001_T1w.nii"
test_load = nib.load(example_filename).get_fdata()
test_load.shape
test = test_load[:, :, 50]
plt.imshow(test)
plt.show()
