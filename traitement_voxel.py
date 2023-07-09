import math
import threading
from pathlib import Path

import nibabel as nb
import numpy as np
import SimpleITK as sitk
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
from scipy.stats import pearsonr


def load_img(filename):
    return nb.load(filename)


def average_intensity(data):
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


def calculate_skewness_kurtosis_of_highest_peak(image_data):
    # Initialize an array to store the skewness for each voxel
    skewness_array = np.zeros(image_data.shape[:-1])
    kurtosis_array = np.zeros(image_data.shape[:-1])

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
                    kurtosis_array[i, j, k] = kurtosis(peak_heights)

    return skewness_array, kurtosis_array


"""
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
"""


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
    local_maxima = argrelextrema(voxel_timeseries, np.greater)[0]
    # Calculate the y-values of the local maxima
    peak_intensities = voxel_timeseries[local_maxima]

    # Calculate the average intensity
    return np.nanmean(peak_intensities)


def calculate_average_peak_intensities(image_data):
    # Initialize an array to store the average intensity for each voxel
    avg_intensity_array = np.zeros(image_data.shape[:-1])

    # Iterate over each voxel
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            for k in range(image_data.shape[2]):
                # Get the timeseries for the current voxel
                voxel_timeseries = image_data[i, j, k, :]
                # Check if the timeseries is not empty
                if ~np.all(voxel_timeseries == voxel_timeseries[0]):
                    # Calculate the average peak intensity
                    avg_intensity = calculate_average_peak_intensity(voxel_timeseries)

                    # Store the average intensity for the current voxel
                    avg_intensity_array[i, j, k] = avg_intensity
                else:
                    # Set NaN for empty timeseries
                    avg_intensity_array[i, j, k] = np.nan

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
                if np.any(voxel_timeseries != voxel_timeseries[0]):
                    # Calculate the standard deviation of the peak intensity
                    std_intensity = calculate_std_peak_intensity(voxel_timeseries)

                    # Store the standard deviation for the current voxel
                    std_intensity_array[i, j, k] = std_intensity
                else:
                    std_intensity_array[i, j, k] = np.nan

    return std_intensity_array


traitement_voxel_result = {}


def traitement_voxel(input_irmf):
    result = {}

    img = load_img(input_irmf)
    data = img.get_fdata()

    seed_point = (10, 50, 50)
    #TURN IMAGE
    lower_threshold = 100
    upper_threshold = 200
    image = sitk.ReadImage(input_irmf)

    segmented_images = []
    for volume in range(0, 341):
        volume_image = image[..., volume]  # Extract the volume

        segmented_volume = region_growing(volume_image, seed_point, lower_threshold, upper_threshold)
        segmented_array = sitk.GetArrayFromImage(segmented_volume)
        segmented_images.append(segmented_array)

    segmented_images = np.array(segmented_images)
    data = segmented_images

    # draw_3d_volumes(data[:, :, :, 10])
    tic_init = time.perf_counter()
    print("--------------")
    max_y_value = max_y(data)
    result["max_y_value"] = max_y_value
    toc = time.perf_counter()
    print(f"Finish max_y in {toc - tic_init:0.4f} seconds")

    tic = time.perf_counter()
    min_y_value = min_y(data)
    result["min_y_value"] = min_y_value
    toc = time.perf_counter()
    print(f"Finish min_y in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    average_intensity_value = average_intensity(data)
    result["average_intensity_value"] = average_intensity_value
    toc = time.perf_counter()
    print(f"Finish average_intensity_value in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    std_deviation = standard_deviation(data)
    result["std_deviation"] = std_deviation
    toc = time.perf_counter()
    print(f"Finish std_deviation in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    skewness = skew(data, axis=3)
    result["skewness"] = skewness
    toc = time.perf_counter()
    print(f"Finish skewness in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    kurtosis_value = kurtosis(data, axis=3)
    result["kurtosis_value"] = kurtosis_value

    toc = time.perf_counter()
    print(f"Finish kurtosis in {toc - tic:0.4f} seconds")

    # Calculate the local maxima for each voxel
    tic = time.perf_counter()
    local_maxima_count = calculate_local_maxima(data)
    result["local_maxima_count"] = local_maxima_count
    toc = time.perf_counter()
    print(f"Finish local_maxima_count in {toc - tic:0.4f} seconds")

    # Calculate the peaks per timeframe
    tic = time.perf_counter()
    peaks_per_timeframe = calculate_peaks_per_timeframe(data, local_maxima_count)
    result["peaks_per_timeframe"] = peaks_per_timeframe
    toc = time.perf_counter()
    print(f"Finish peaks_per_timeframe in {toc - tic:0.4f} seconds")

    # OK
    tic = time.perf_counter()
    skewness_of_highest_peak, kurtosis_of_highest_peak = calculate_skewness_kurtosis_of_highest_peak(data)
    result["skewness_of_highest_peak"] = skewness_of_highest_peak
    result["kurtosis_of_highest_peak"] = kurtosis_of_highest_peak
    toc = time.perf_counter()
    print(f"Finish skewness_of_highest_peak and kurtosis ok in {toc - tic:0.4f} seconds")

    # Calculate the standard deviation of the peak intervals for each voxel
    tic = time.perf_counter()
    std_of_peak_intervals = calculate_std_of_peak_intervals(data)
    result["std_of_peak_intervals"] = std_of_peak_intervals
    toc = time.perf_counter()
    print(f"Finish std_of_peak_intervals in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    average_peak_intensities = calculate_average_peak_intensities(data)
    result["average_peak_intensities"] = average_peak_intensities
    toc = time.perf_counter()
    print(f"Finish average_peak_intensities in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    std_peak_intensities = calculate_std_peak_intensities(data)
    result["std_peak_intensities"] = std_peak_intensities
    toc = time.perf_counter()
    print(f"Finish std_peak_intensities in {toc - tic:0.4f} seconds")

    print("CALCUL FINISHED")
    traitement_voxel_result[input_irmf.name] = result


def region_growing(image, seed_point, lower_threshold, upper_threshold):
    # Create a binary mask to store the segmented region
    segmented_region = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    segmented_region.CopyInformation(image)
    segmented_region_array = sitk.GetArrayViewFromImage(segmented_region)
    # Create a queue to store the neighboring points
    seed_queue = []
    seed_queue.append(seed_point)
    # Perform region growing
    while seed_queue:
        current_point = seed_queue.pop(0)
        current_value = image.GetPixel(current_point[::-1])
        if segmented_region_array[current_point] == 0 and lower_threshold <= current_value <= upper_threshold:
            segmented_region_array[current_point] = 1

            # Get neighboring points
            neighbors = [(current_point[0] + 1, current_point[1], current_point[2]),
                         (current_point[0] - 1, current_point[1], current_point[2]),
                         (current_point[0], current_point[1] + 1, current_point[2]),
                         (current_point[0], current_point[1] - 1, current_point[2]),
                         (current_point[0], current_point[1], current_point[2] + 1),
                         (current_point[0], current_point[1], current_point[2] - 1)]

            # Add neighboring points to the queue
            for neighbor in neighbors:
                if segmented_region_array[neighbor] == 0:
                    seed_queue.append(neighbor)

    return segmented_region


def writeResult(result):
    with h5py.File('results_growing.hdf5', 'w') as f:
        grp = f.create_group(file_nii.name)

        grp.create_dataset('max_y_value', data=result["max_y_value"])
        grp.create_dataset('min_y_value', data=result["min_y_value"])
        grp.create_dataset('average_intensity_value', data=result["average_intensity_value"])
        grp.create_dataset('std_deviation', data=result["std_deviation"])
        grp.create_dataset('skewness', data=result["skewness"])
        grp.create_dataset('kurtosis_value', data=result["kurtosis_value"])
        grp.create_dataset('local_maxima_count', data=result["local_maxima_count"])
        grp.create_dataset('skewness_of_highest_peak', data=result["skewness_of_highest_peak"])
        grp.create_dataset('kurtosis_of_highest_peak', data=result["kurtosis_of_highest_peak"])
        grp.create_dataset('std_of_peak_intervals', data=result["std_of_peak_intervals"])
        grp.create_dataset('average_peak_intensities', data=result["average_peak_intensities"])
        grp.create_dataset('std_peak_intensities', data=result["std_peak_intensities"])


if __name__ == "__main__":
    sclice_timing_correction_threads = []

    threads = []
    for file_nii in Path("mc_flirt").glob("**/*.gz"):
        if "Cyberball" in file_nii.name:
            # Lance tous les threads
            thread = threading.Thread(target=traitement_voxel, args=(file_nii,))
            threads.append(thread)

    for thread in threads:
        thread.start()

    # Attend la fin de tous les threads
    for thread in threads:
        thread.join()

    for key, value in traitement_voxel_result.items():
        print("WRITE " + key)
        writeResult(value)
