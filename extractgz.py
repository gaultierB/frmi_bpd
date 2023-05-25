import gzip
import shutil


def extract_gz_file(input_file_path, output_file_path):
    with gzip.open(input_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


for i in range(1, 40):
    if i < 10:
        i = '01'
    name_file = 'sub-EESS0'+ str(i)
    output_file = '.nii'  # Chemin vers le fichier de sortie extrait
    extract_gz_file(input_file, output_file)
