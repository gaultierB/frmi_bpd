import gzip
import shutil
import csv
import os


def extract_gz_file(input_file_path, output_file_path):
    with gzip.open(input_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def get_list_filename():
    output_files_Cyberball = []
    output_files_T1w = []
    output_folders = []
    participants_file = 'ds000214-download/participants.tsv'  # Chemin vers le fichier participants.tsv
    with open(participants_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Ignorer la première ligne d'en-tête
        for row in reader:
            participant_name = "/sub-" + row[0]
            output_folder = "data/" + row[3] + participant_name
            output_folders.append(output_folder)
            output_file = output_folder + participant_name + '_T1w.nii'  # Chemin vers le fichier de sortie extrait
            output_files_T1w.append(output_file)
            output_file = output_folder + participant_name + '_task-Cyberball_bold.nii'  # Chemin vers le fichier de sortie extrait
            output_files_Cyberball.append(output_file)
    return output_files_Cyberball, output_files_T1w, output_folders


if __name__ == '__main__':
    participants_file = 'ds000214-download/participants.tsv'  # Chemin vers le fichier participants.tsv

    with open(participants_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Ignorer la première ligne d'en-tête

        for row in reader:
            participant_name = "/sub-" + row[0]
            input_file = "ds000214-download" + participant_name + '/anat' + participant_name + '_T1w.nii' + '.gz'  # Construire le chemin du fichier gzip
            output_folder = "data/" + row[3] + participant_name
            os.makedirs(output_folder, exist_ok=True)  # Créer le dossier s'il n'existe pas déjà
            output_file = output_folder + participant_name + '_T1w.nii'  # Chemin vers le fichier de sortie extrait
            extract_gz_file(input_file, output_file)

            input_file = "ds000214-download" + participant_name + '/func' + participant_name + '_task-Cyberball_bold.nii' + '.gz'  # Construire le chemin du fichier gzip
            output_file = output_folder + participant_name + '_task-Cyberball_bold.nii'  # Chemin vers le fichier de sortie extrait
            extract_gz_file(input_file, output_file)
