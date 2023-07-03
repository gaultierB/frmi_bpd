import os
import time
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import extractgz
import matplotlib.pyplot as plt
from fsl.wrappers import bet
from fsl.wrappers import fslmaths
import subprocess
import json
import traitement_voxel
import threading

# https://socr.umich.edu/HTML5/BrainViewer/

def read_mean_image(filename):
    image = sitk.ReadImage(filename)
    image_array = sitk.GetArrayViewFromImage(image)

    print('Image shape:', image_array.shape)
    print('Image data type:', image_array.dtype)

    mean_intensity = np.mean(image_array)
    print('Mean intensity:', mean_intensity)
    return mean_intensity


def mean_intensity(output_files_Cyberball):
    control_mean = []
    patient_mean = []

    for filename in output_files_Cyberball:
        if "Control" in filename:
            control_mean.append(read_mean_image(filename))
        else:
            patient_mean.append(read_mean_image(filename))

    max_length = max(len(control_mean), len(patient_mean))
    control_mean += [0] * (max_length - len(control_mean))
    patient_mean += [0] * (max_length - len(patient_mean))

    mean_intensity(control_mean, patient_mean)

    # Création des positions des barres sur l'axe x
    x = np.arange(len(control_mean))
    # Largeur des barres
    width = 0.35
    # Création de la figure et des axes
    fig, ax = plt.subplots()
    # Création des barres pour les moyennes de contrôle
    bars1 = ax.bar(x - width / 2, control_mean, width, label='Contrôle')
    # Création des barres pour les moyennes de patients
    bars2 = ax.bar(x + width / 2, patient_mean, width, label='Patients')
    # Ajout des étiquettes des groupes sur l'axe x
    ax.set_xticks(x)
    # Ajout d'une légende
    ax.legend()
    # Ajout d'un titre
    ax.set_title('Comparaison des moyennes')
    # Affichage du graphique
    plt.show()


def BET(input_file):
    bet = "bet"
    os.makedirs(bet, exist_ok=True)  # Create the folder if it doesn't exist
    output_image = Path(bet, input_file.name)

    command = ['bet', input_file, output_image, "-F"]
    subprocess.run(command)
    print("BET sont terminées.")

    return output_image


def run_mcflirt(input_file):
    mc_flirt = "mc_flirt"
    os.makedirs(mc_flirt, exist_ok=True)  # Create the folder if it doesn't exist
    output_image = Path(mc_flirt, input_file.name)


    # Execute MCFLIRT
    command = ['mcflirt', '-in', input_file, '-out', output_image]
    subprocess.run(command)

def sclice_timing_correction(input_file):
    folder_spacial_smoothing = "sclice_timing_correction"
    os.makedirs(folder_spacial_smoothing, exist_ok=True)  # Create the folder if it doesn't exist
    output_image = Path(folder_spacial_smoothing, file_nii.name)

    print("---OUTPUT---")
    print(output_image.resolve())
    print("------")

    tr = 3.0  # Temps de répétition (TR) en secondes
    ta = 1.0  # Temps d'acquisition (TA) de chaque tranche en secondes
    interleaved = True  # True si vos tranches sont acquises de manière entrelacée, False sinon

    # Construisez la commande FSL pour la correction de synchronisation des tranches
    slice_timing_correction_config = Path("ds000214-download", "slice_timing_correction.txt")
    cmd = f"slicetimer -i {input_file} -o {output_image} --tcustom=" + str(slice_timing_correction_config.resolve())
    if interleaved:
        cmd += " --odd"

    # Exécutez la commande FSL
    subprocess.call(cmd, shell=True)

    print("synchronisation des tranches terminées.")
    return output_image


# /mnt/c/Users/gault/PycharmProjects/frmi_bpd/datasetFSL/Control/IMAGE

def spacial_smoothing(file_nii):
    folder_spacial_smoothing = "spacial_smoothing"
    os.makedirs(folder_spacial_smoothing, exist_ok=True)  # Create the folder if it doesn't exist
    output_image = Path(folder_spacial_smoothing, file_nii.name)
    # Taille de la fenêtre de lissage (en mm)
    smoothing_size = 5

    # Commande de lissage spatial avec FSL
    fsl_cmd = f"fslmaths {file_nii} -s {smoothing_size} {output_image}"

    # Exécution de la commande
    subprocess.call(fsl_cmd, shell=True)
    # Exécutez la commande FSL
    print("spacial_smoothing des tranches terminées.")
    return output_image


def intensity_normalization(input_file):
    folder_intensity_normalization = "intensity_normalization"
    os.makedirs(folder_intensity_normalization, exist_ok=True)  # Create the folder if it doesn't exist
    output_image = Path(folder_intensity_normalization, file_nii.name)

    mean_value = subprocess.check_output("fslstats " + str(input_file) + " -M", shell=True).decode().strip()
    standard_deviation = subprocess.check_output("fslstats " + str(input_file) + " -s", shell=True).decode().strip()

    subprocess.call(
        "fslmaths " + str(input_file) + " -sub " + str(mean_value) + " -div " + str(standard_deviation) + " " + str(output_image),
        shell=True)
    print("intensity_normalization des tranches terminées.")
    return output_image


if __name__ == "__main__":
    tic = time.perf_counter()

    threads = []

    list_Path_ = []
    for file_nii in Path("datasetFSL").glob("**/*.nii"):
        if "Cyberball" in file_nii.name:
            thread = threading.Thread(target=sclice_timing_correction, args=(file_nii,))
            threads.append(thread)
            break


    for file_nii in Path("sclice_timing_correction").glob("*nii.gz"):
        if "Cyberball" in file_nii.name:
            thread = threading.Thread(target=BET, args=(file_nii,))
            threads.append(thread)
            break

    for file_nii in Path("bet").glob("**/*.gz"):
        if "Cyberball" in file_nii.name:
            thread = threading.Thread(target=spacial_smoothing, args=(file_nii,))
            threads.append(thread)
            break

    for file_nii in Path("spacial_smoothing").glob("**/*.gz"):
        if "Cyberball" in file_nii.name:
            thread = threading.Thread(target=intensity_normalization, args=(file_nii,))
            threads.append(thread)
            break

    for file_nii in Path("intensity_normalization").glob("**/*.gz"):
        if "Cyberball" in file_nii.name:
            thread = threading.Thread(target=run_mcflirt, args=(file_nii,))
            run_mcflirt(file_nii)
            break

    for file_nii in Path("mc_flirt").glob("**/*.gz"):
        if "Cyberball" in file_nii.name:
            traitement_voxel.traitement_voxel(file_nii)
            thread = threading.Thread(target=BET, args=(file_nii,))
            threads.append(thread)
            break

    # Lance tous les threads
    for thread in threads:
        thread.start()

    # Attend la fin de tous les threads
    for thread in threads:
        thread.join()


    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

# export FSLDIR=/usr/local/fsl
# source /usr/local/fsl/etc/fslconf/fsl.sh
