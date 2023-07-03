import math
import multiprocessing
from multiprocessing import Pool

import nibabel as nb
import numpy as np
from scipy.stats import norm, kurtosis
from scipy.stats import skew
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage.filters import maximum_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nibabel as nib
from scipy.signal import argrelextrema
import time
import h5py

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


def draw_3d_volumes(data):
    # Création de la figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z_values = data[:, 2]  # Valeurs de l'axe Z

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Création du nuage de points avec différentes couleurs basées sur l'axe Z
    cmap = plt.cm.get_cmap('viridis')  # Choix de la colormap (ici 'viridis', mais vous pouvez en choisir une autre)
    normalize = plt.Normalize(vmin=z.min(), vmax=z.max())  # Normalisation des valeurs de Z pour la colormap
    # Création du nuage de points avec les couleurs spécifiées
    ax.scatter(x, y, z, c=z_values, cmap=cmap, norm=normalize, marker='o')

    # Paramètres d'affichage
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')

    # Affichage de la figure
    plt.show()


def calculate_local_maxima(image_data):
    # Initialize an array to store the count of local maxima for each voxel
    local_maxima_count = np.zeros(image_data.shape[:-1])

    # Iterate over each voxel
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            for k in range(image_data.shape[2]):
                # Get the timeseries for the current voxel
                voxel_timeseries = image_data[i, j, k, :]

                # Calculate the local maxima
                local_maxima = argrelextrema(voxel_timeseries, np.greater)

                # Store the count of local maxima for the current voxel
                local_maxima_count[i, j, k] = len(local_maxima[0])

    return local_maxima_count


def calculate_peaks_per_timeframe(image_data, local_maxima_count):
    timeframes = image_data.shape[-1]

    # Calculate the total peaks per timeframe
    peaks_per_timeframe = np.sum(local_maxima_count) / timeframes

    return peaks_per_timeframe


def calculate_peak_heights(voxel_timeseries):
    # Calculate the local maxima and minima
    local_maxima = argrelextrema(voxel_timeseries, np.greater)[0]
    local_minima = argrelextrema(voxel_timeseries, np.less)[0]

    peak_heights = []

    # Iterate over each local maxima
    for i in range(len(local_maxima)):
        # Find the neighbouring local minima for each local maxima
        left_minima = local_minima[local_minima < local_maxima[i]]
        right_minima = local_minima[local_minima > local_maxima[i]]

        # If there is no left or right local minima, continue to the next local maxima
        if len(left_minima) == 0 or len(right_minima) == 0:
            continue

        # Calculate the y-value difference with the left and right local minima
        left_difference = voxel_timeseries[local_maxima[i]] - voxel_timeseries[left_minima[-1]]
        right_difference = voxel_timeseries[local_maxima[i]] - voxel_timeseries[right_minima[0]]

        # Calculate the height of the peak and append it to the list
        peak_height = (left_difference + right_difference) / 2
        peak_heights.append(peak_height)

    return np.array(peak_heights)


def calculate_skewness_of_highest_peak(image_data):
    # Initialize an array to store the skewness for each voxel
    skewness_array = np.zeros(image_data.shape[:-1])

    # Iterate over each voxel
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            for k in range(image_data.shape[2]):
                # Get the timeseries for the current voxel
                voxel_timeseries = image_data[i, j, k, :]

                # Calculate the peak heights
                peak_heights = calculate_peak_heights(voxel_timeseries)

                if len(peak_heights) > 0:
                    # Calculate the skewness of the highest peak
                    skewness_array[i, j, k] = skew(peak_heights)

    return skewness_array


def calculate_kurtosis_of_highest_peak(image_data):
    # Initialize an array to store the skewness for each voxel
    skewness_array = np.zeros(image_data.shape[:-1])

    # Iterate over each voxel
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            for k in range(image_data.shape[2]):
                # Get the timeseries for the current voxel
                voxel_timeseries = image_data[i, j, k, :]

                # Calculate the peak heights
                peak_heights = calculate_peak_heights(voxel_timeseries)

                if len(peak_heights) > 0:
                    # Calculate the kurtosis of the highest peak
                    skewness_array[i, j, k] = kurtosis(peak_heights)

    return skewness_array


def calculate_peak_intervals(voxel_timeseries):
    # Calculate the local maxima
    local_maxima = argrelextrema(voxel_timeseries, np.greater)[0]

    # Calculate the intervals between the local maxima
    peak_intervals = np.diff(local_maxima)

    return peak_intervals


def calculate_std_of_peak_intervals(image_data):
    # Initialize an array to store the standard deviation for each voxel
    std_array = np.zeros(image_data.shape[:-1])

    # Iterate over each voxel
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            for k in range(image_data.shape[2]):
                # Get the timeseries for the current voxel
                voxel_timeseries = image_data[i, j, k, :]

                # Calculate the peak intervals
                peak_intervals = calculate_peak_intervals(voxel_timeseries)

                if len(peak_intervals) > 0:
                    # Calculate the standard deviation of the peak intervals
                    std_array[i, j, k] = np.std(peak_intervals)

    return std_array


def calculate_average_peak_intensity(voxel_timeseries):
    # Calculate the local maxima
    local_maxima = argrelextrema(voxel_timeseries, np.greater)[0]

    # Calculate the y-values of the local maxima
    peak_intensities = voxel_timeseries[local_maxima]

    # Calculate the average intensity
    average_intensity = np.mean(peak_intensities)

    return average_intensity


def calculate_average_peak_intensities(image_data):
    # Initialize an array to store the average intensity for each voxel
    avg_intensity_array = np.zeros(image_data.shape[:-1])

    # Iterate over each voxellinke
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            for k in range(image_data.shape[2]):
                # Get the timeseries for the current voxel
                voxel_timeseries = image_data[i, j, k, :]

                # Calculate the average peak intensity
                avg_intensity = calculate_average_peak_intensity(voxel_timeseries)

                # Store the average intensity for the current voxel
                avg_intensity_array[i, j, k] = avg_intensity

    return avg_intensity_array


def calculate_std_peak_intensity(voxel_timeseries):
    # Calculate the local maxima
    local_maxima = argrelextrema(voxel_timeseries, np.greater)[0]

    # Calculate the y-values of the local maxima
    peak_intensities = voxel_timeseries[local_maxima]

    # Calculate the standard deviation of the intensities
    std_intensity = np.std(peak_intensities)

    return std_intensity


def calculate_std_peak_intensities(image_data):
    # Initialize an array to store the standard deviation for each voxel
    std_intensity_array = np.zeros(image_data.shape[:-1])

    # Iterate over each voxel
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            for k in range(image_data.shape[2]):
                # Get the timeseries for the current voxel
                voxel_timeseries = image_data[i, j, k, :]

                # Calculate the standard deviation of the peak intensity
                std_intensity = calculate_std_peak_intensity(voxel_timeseries)

                # Store the standard deviation for the current voxel
                std_intensity_array[i, j, k] = std_intensity

    return std_intensity_array

def execute_function(args):
    func, data = args
    return func(data)


def traitement_voxel(input_irmf):
    img = load_img(input_irmf)
    data = img.get_fdata()
    functions = [
        (max_y, {}),
        (min_y, {}),
        (average_intensity, {}),
        (standard_deviation, {}),
        (skew, {'axis': 3}),
        (kurtosis, {'axis': 3}),
        (calculate_local_maxima, {}),
        (calculate_skewness_of_highest_peak, {}),
        (calculate_kurtosis_of_highest_peak, {}),
        (calculate_std_of_peak_intervals, {}),
        (calculate_average_peak_intensities, {}),
        (calculate_std_peak_intensities, {})
    ]

    # Create a multiprocessing Pool
    with Pool(multiprocessing.cpu_count()) as p:
        results = p.map(execute_function, [(func, data, kwargs) for func, kwargs in functions])

    # Unpack results
    max_y_value, min_y_value, average_intensity_value, std_deviation, skewness, e, local_maxima_count, skewness_of_highest_peak, kurtosis_of_highest_peak, std_of_peak_intervals, average_peak_intensities, std_peak_intensities = results

    with h5py.File('results.hdf5', 'w') as f:
        f.create_dataset('max_y_value', data=max_y_value)
        f.create_dataset('min_y_value', data=min_y_value)
        f.create_dataset('average_intensity_value', data=average_intensity_value)
        f.create_dataset('std_deviation', data=std_deviation)
        f.create_dataset('skewness', data=skewness)
        f.create_dataset('e', data=e)
        f.create_dataset('local_maxima_count', data=local_maxima_count)
        f.create_dataset('skewness_of_highest_peak', data=skewness_of_highest_peak)
        f.create_dataset('kurtosis_of_highest_peak', data=kurtosis_of_highest_peak)
        f.create_dataset('std_of_peak_intervals', data=std_of_peak_intervals)
        f.create_dataset('average_peak_intensities', data=average_peak_intensities)
        f.create_dataset('std_peak_intensities', data=std_peak_intensities)

    #READ DATA
#    with h5py.File('results.hdf5', 'r') as f:
#        max_y_value = f['max_y_value'][:]
#        min_y_value = f['min_y_value'][:]