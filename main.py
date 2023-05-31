import os

import SimpleITK as sitk
import numpy as np
import extractgz
import matplotlib.pyplot as plt
from fsl.wrappers import bet

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
        filename = path_file.replace(folder, "")[1:]
        output_image = folder + "BET" + path_file.replace(folder, "")
        os.makedirs(folder, exist_ok=True)  # Créer le dossier s'il n'existe pas déjà
        # Paramètres BET
        frac = 0.5  # Fraction d'intensité pour BET (0.1 à 0.9)
        # Exécuter BET
        print(output_image)
        bet_cmd = bet(path_file, output_image)
        bet_cmd.inputs.in_file = path_file
        bet_cmd.inputs.out_file = output_image
        bet_cmd.run()

    print("Le cerveau a été extrait avec succès et enregistré dans : {}".format(output_image))


if __name__ == "__main__":
    output_files_Cyberball, output_files_T1w, output_folders BET= extractgz.get_list_filename()
    print(output_folders, output_files_T1w)
    BET(output_folders, output_files_T1w)


# https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
# https://neuraldatascience.io/8-mri/nifti.html
# https://medium.com/miccai-educational-initiative/how-to-get-started-with-deep-learning-using-mri-data-5d6a41dbc417
