import os

import SimpleITK as sitk
import numpy as np
import extractgz
import matplotlib.pyplot as plt
from fsl.wrappers import bet
from fsl.wrappers import fslmaths
import subprocess


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


def BET(folders, output_files_T1w):
    for folder, path_file in zip(folders, output_files_T1w):
        output_image = "/mnt/c/Users/gault/PycharmProjects/frmi_bpd/" + folder + "/BET" + path_file.replace(folder, "")
        os.makedirs(folder, exist_ok=True)  # Créer le dossier s'il n'existe pas déjà
        # Paramètres BET
        frac = 0.5  # Fraction d'intensité pour BET (0.1 à 0.9)
        # Exécuter BET
        print(output_image)
        bet_cmd = bet(path_file, output_image)
        print("Le cerveau a été extrait avec succès et enregistré dans : {}".format(output_image))


def run_mcflirt(folders, output_files_T1w):
    for folder, path_file in zip(folders, output_files_T1w):
        output_image = "/mnt/c/Users/gault/PycharmProjects/frmi_bpd/" + folder + "/MCFLIRT" + path_file.replace(folder, "")
        folder = os.path.dirname(output_image)
        os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
        # Execute MCFLIRT
        command = ['mcflirt', '-in', path_file, '-out', output_image]
        subprocess.run(command)
        print("MCFLIRT successfully applied and saved the output to: {}".format(output_image))


def sclice_timing_correction(input_files, folders):
    tr = 3.0  # Temps de répétition (TR) en secondes
    ta = 1.0  # Temps d'acquisition (TA) de chaque tranche en secondes
    # num_slices = 30  # Nombre de tranches dans vos images IRMF
    interleaved = True  # True si vos tranches sont acquises de manière entrelacée, False sinon

    # Parcourez chaque fichier d'IRMF et appliquez la correction de synchronisation des tranches
    for input_file, output_folder in zip(input_files, folders):
        output_path = output_folder + "/SCLICE_TC" + input_file.replace(output_folder, "")
        os.makedirs(output_path, exist_ok=True)  # Create the folder if it doesn't exist
        # Construisez la commande FSL pour la correction de synchronisation des tranches
        cmd = f"slicetimer -i {input_file} -o {output_path} --odd --ocustom={output_path} --tcustom={tr} --repeat={ta}"
        if interleaved:
            cmd += " --odd"

        # Exécutez la commande FSL
        subprocess.call(cmd, shell=True)

        print(f"Correction de synchronisation des tranches terminée pour {input_file}")

    print("Toutes les corrections de synchronisation des tranches sont terminées.")


def spacial_smoothing(input_files, folders):
    for input_file, output_folder in zip(input_files, folders):
        # Nom du fichier de sortie
        output_path = output_folder + "/SCLICE_TC" + input_file.replace(output_folder, "")
        output_file = "image_smoothed.nii.gz"

        # Taille de la fenêtre de lissage (en mm)
        smoothing_size = 5

        # Commande de lissage spatial avec FSL
        # fsl_cmd = f"fslmaths {os.path.join(data_folder, input_file)} -s {smoothing_size} {os.path.join(data_folder, output_file)}"

        # Exécution de la commande
        # subprocess.call(fsl_cmd, shell=True)


if __name__ == "__main__":
    # TODO analyse shape pour savoir qui est le IRM et l'IRMF
    # Refaire les traitements sur le shape de 4

    output_files_Cyberball, output_files_T1w, folders = extractgz.get_list_filename()
    print(output_files_Cyberball)
    print(output_files_T1w)
    print(folders)
    # BET(output_folders, output_files_T1w)
    # run_mcflirt(output_folders, output_files_T1w)
    sclice_timing_correction(output_files_T1w, folders)

# https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
# https://neuraldatascience.io/8-mri/nifti.html
# https://medium.com/miccai-educational-initiative/how-to-get-started-with-deep-learning-using-mri-data-5d6a41dbc417


# export FSLDIR=/usr/local/fsl
# source /usr/local/fsl/etc/fslconf/fsl.sh
