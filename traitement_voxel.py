import math

import nibabel as nb
import numpy as np
from scipy.stats import norm, kurtosis
from scipy.stats import skew
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage.filters import maximum_filter


def load_img(filename):
    return nb.load(filename)


def average_intensity(data):
    print(data.shape)
    n = data.shape[3]  # Nombre de timepoints nous avons dans notre dataset 341 volume d'image
    sum_y = np.sum(data, axis=3)  # Somme des valeurs d'y pour chaque voxel
    average_intensity = sum_y / n  # Calcul de la moyenne des valeurs d'y
    return average_intensity


def standard_deviation(data):
    return np.std(data, axis=3)


def max_y(data):
    return np.max(data, axis=3)


def min_y(data):
    return np.min(data, axis=3)

if __name__ == '__main__':
    filename = "C:\\Users\\gault\\PycharmProjects\\frmi_bpd\\data\\Control\\sub-EESS001\\sub-EESS001_task-Cyberball_bold.nii"
    img = load_img(filename)
    data = img.get_fdata()
    a = max_y(data)
    print(a.shape)
    b = min_y(data)
    c = average_intensity(data)
    d = standard_deviation(data)
    skewness = skew(data, axis=3)
    e = kurtosis(data, axis=3)
