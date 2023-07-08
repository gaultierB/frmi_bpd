import h5py
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # Ouvrir le fichier HDF5
    file = h5py.File('results.hdf5', 'r')

    # Liste des clés du dataset
    keys = list(file.keys())

    # Créer une liste vide pour stocker les données de chaque clé
    data_list = []

    with h5py.File('results.hdf5', 'r') as f:
        print(f)
        # Lire les données
        max_y_value = np.array(f['max_y_value'])
        min_y_value = np.array(f['min_y_value'])
        max_y_value = max_y_value.flatten()
        min_y_value = min_y_value.flatten()
            # ... et ainsi de suite pour les autres jeux de données

        # Créer un nouvel histogramme pour max_y_value
        plt.figure(figsize=(10, 6))
        plt.hist(max_y_value, bins=30, alpha=0.5, label='max_y_value')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of max_y_value')
        plt.legend(loc='upper right')
        plt.savefig("histo.png")

    # Fermer le fichier HDF5
    file.close()