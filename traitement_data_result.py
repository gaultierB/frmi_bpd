import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Ouvrir le fichier HDF5
    with h5py.File('results.hdf5', 'r') as file:

        # Parcourir chaque groupe
        for group_name in file.keys():
            group = file[group_name]

            # Parcourir chaque clé dans le groupe
            for key_name in group.keys():
                data = np.array(group[key_name])
                print(data)
                print(data[45][34][10])
                # Calculer la matrice de corrélation
                correlation_matrix = np.corrcoef(data[45][34][10], rowvar=False)

                # Afficher la matrice de corrélation
                plt.figure(figsize=(10, 6))
                plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Correlation Matrix of {key_name} in {group_name}')
                plt.show()
