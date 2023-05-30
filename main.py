import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_frmi(filename):
    test_load = nib.load(filename).get_fdata()
    print(test_load.shape)
    test = test_load[:, :, 50]
    plt.imshow(test)
    plt.show()

example_filename = "data/Patient/sub-EESS002/sub-EESS002_T1w.nii"
cyberball = "data/Patient/sub-EESS002/sub-EESS002_task-Cyberball_bold.nii"
load_frmi(cyberball)
#https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
#https://neuraldatascience.io/8-mri/nifti.html
#https://medium.com/miccai-educational-initiative/how-to-get-started-with-deep-learning-using-mri-data-5d6a41dbc417