import gzip
import shutil
import csv


def extract_gz_file(input_file_path, output_file_path):
    with gzip.open(input_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    participants_file = 'ds000214-download/participants.tsv'  # Chemin vers le fichier participants.tsv

    with open(participants_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Ignorer la première ligne d'en-tête

        for row in reader:
            participant_name = "sub-" + row[0]
            input_file = participant_name + '/anat/' + participant_name + '_T1w.nii' + '.gz'  # Construire le chemin du fichier gzip
            output_file = row[3] + '/' + participant_name + 'TEZST.nii'  # Chemin vers le fichier de sortie extrait
            extract_gz_file(input_file, output_file)
