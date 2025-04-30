import os
import h5py
import numpy as np
import scipy.io as sio

#
datasets = [
    'fused_Ag', 'fused_Au', 'fused_Bi', 'fused_Co', 'fused_Cu',
    'fused_In', 'fused_Ni', 'fused_Pd', 'fused_Pt', 'fused_Rh',
    'fused_Ru', 'fused_haadf'
]

# datasets = [
#     'raw_Ag', 'raw_Au', 'raw_Bi', 'raw_Co', 'raw_Cu',
#     'raw_In', 'raw_Ni', 'raw_Pd', 'raw_Pt', 'raw_Rh',
#     'raw_Ru', 'raw_haadf'
# ]

#
data_dict = {dataset: [] for dataset in datasets}

#
for layer in range(1, 12):
    folder_name = f'layer_{layer}'
    h5_file_path = os.path.join(folder_name, 'Fused_Reconstruction.h5')

    #
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")

    #
    with h5py.File(h5_file_path, 'r') as f:
        for dataset in datasets:

            data_2d = f[dataset][:]
            data_dict[dataset].append(data_2d)

#
for dataset in datasets:
    #
    data_3d = np.stack(data_dict[dataset], axis=-1)

    #
    mat_file_name = f'projs_{dataset}.mat'
    sio.savemat(mat_file_name, {dataset: data_3d})

    print(f"Saved {mat_file_name} with shape {data_3d.shape}")