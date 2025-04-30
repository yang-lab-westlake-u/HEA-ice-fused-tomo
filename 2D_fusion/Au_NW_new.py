import fusion_utils as utils
from scipy.sparse import spdiags
import sparse
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import h5py
import scipy.io as sio
from skimage.io import imsave
from skimage import exposure, img_as_uint

#
haadf_data_file = 'input/AuHEANW_241007t2_EDS_projs_align_241028.mat'
chem_data_file = 'input/AuHEANW_241007t2_EDS_mapping_align_norm_241231_new.mat'

#
haadf_mat = sio.loadmat(haadf_data_file)
HAADF_all = haadf_mat['EDS_projs_new_n']

#
chem_mat = sio.loadmat(chem_data_file)
chem_maps = chem_mat['element_align'][0, 0]

#
elem_names = ['Co', 'Ni', 'Cu', 'Ru', 'Rh', 'Pd', 'Ag', 'In', 'Pt', 'Au', 'Bi']
elem_weights = [27, 28, 29, 44, 45, 46, 47, 49, 78, 79, 83]

#
#
chem_data = {}
for elem in elem_names:
    field_name = elem + '_align'
    if field_name in chem_maps.dtype.names:  #
        chem_data[elem] = chem_maps[field_name]  #
    else:
        raise ValueError(f"Field {field_name} not found in chem_maps")

#
nz = len(elem_names)
lambdaHAADF = 1 / nz
gamma = 1.6
nIter = 30  # Typically 10-15 will suffice
#lambdaTV = 0.022  # Typically between 0.001 and 1
lambdaTV = 0.01
lambdaChem = 0.015
#lambdaChem = -0.01  #between 0.05 and 0.3
bkg = 1e-2
regularize = True
nIter_TV = 6

#
#for layer in range(HAADF_all.shape[2]):
for layer in range(5,6):
    print(f"Processing layer {layer+1} of {HAADF_all.shape[2]}")

    #
    HAADF = HAADF_all[:, :, layer]
    HAADF = np.array(HAADF)
    HAADF -= np.min(HAADF)
    HAADF /= np.max(HAADF)
    HAADF_flat = HAADF.flatten()

    #
    xx = np.array([], dtype=np.float32)
    for elem in elem_names:
        chemMap = chem_data[elem][:, :, layer]  #
        chemMap = np.array(chemMap)
        #chemMap -= np.min(chemMap)
        #chemMap /= np.max(chemMap)
        xx = np.concatenate([xx, chemMap.flatten()])

    #
    xx0 = xx.copy()

    #
    (nx, ny) = HAADF.shape
    nPix = nx * ny

    #
    A = utils.create_weighted_measurement_matrix(nx, ny, nz, elem_weights, gamma, 0)

    # C++ TV Min Regularizers
    reg = utils.tvlib(nx, ny)

    #
    xx = xx0.copy()
    xx = np.where((xx < .2), 0, xx)

    #
    costHAADF = np.zeros(nIter, dtype=np.float32)
    costChem = np.zeros(nIter, dtype=np.float32)
    costTV = np.zeros(nIter, dtype=np.float32)

    lsqFun = lambda inData: 0.5 * np.linalg.norm(A.dot(inData ** gamma) - HAADF_flat) ** 2
    poissonFun = lambda inData: np.sum(xx0 * np.log(inData + 1e-8) - inData)

    for kk in tqdm(range(nIter)):
        xx -= gamma * spdiags(xx ** (gamma - 1), [0], nz * nx * ny, nz * nx * ny) * lambdaHAADF * A.transpose() * (
                A.dot(xx ** gamma) - HAADF_flat) + lambdaChem * (1 - xx0 / (xx + bkg))

        #
        xx[xx < 0] = 0

        #
        if regularize:
            for zz in range(nz):
                xx[zz * nPix:(zz + 1) * nPix] = reg.fgp_tv(xx[zz * nPix:(zz + 1) * nPix].reshape(nx, ny), lambdaTV,
                                                           nIter_TV).flatten()
                costTV[kk] += reg.tv(xx[zz * nPix:(zz + 1) * nPix].reshape(nx, ny))

        costHAADF[kk] = lsqFun(xx)
        costChem[kk] = poissonFun(xx)

    utils.plot_convergence(costHAADF, lambdaHAADF, costChem, lambdaChem, costTV, lambdaTV)

    #
    save_folder_name = f'./output/layer_{layer+1}'
    utils.save_data(save_folder_name, xx0, xx, HAADF, A.dot(xx ** gamma), elem_names, nx, ny, costHAADF, costChem, costTV,
                    lambdaHAADF, lambdaChem, lambdaTV, gamma)

    ## Optional: Plot your raw elastic/inelastic data
    fig, ax = plt.subplots(2, len(elem_names) + 1, figsize=(12, 6.5))
    ax = ax.flatten()

    #
    ax[0].imshow(HAADF.reshape(nx, ny), cmap='gray')
    ax[0].set_title('HAADF')
    ax[0].axis('off')

    #
    ax[1 + len(elem_names)].imshow(HAADF.reshape(nx, ny)[135:235, 200:300], cmap='gray')
    ax[1 + len(elem_names)].set_title('HAADF Cropped')
    ax[1 + len(elem_names)].axis('off')

    #
    for ii in range(len(elem_names)):
        ax[ii + 1].imshow(xx0[ii * (nx * ny):(ii + 1) * (nx * ny)].reshape(nx, ny), cmap='gray')
        ax[ii + 2 + len(elem_names)].imshow(xx0[ii * (nx * ny):(ii + 1) * (nx * ny)].reshape(nx, ny)[135:235, 200:300],
                                            cmap='gray')

        ax[ii + 1].set_title(elem_names[ii])
        ax[ii + 1].axis('off')
        ax[ii + 2 + len(elem_names)].set_title(elem_names[ii] + ' Cropped')
        ax[ii + 2 + len(elem_names)].axis('off')

    fig.tight_layout()
    plt.show()

    ## Plot reconstructed data
    fig, ax = plt.subplots(2, len(elem_names) + 1, figsize=(12, 6.5))
    ax = ax.flatten()

    #
    ax[0].imshow((A.dot(xx ** gamma)).reshape(nx, ny), cmap='gray')
    ax[0].set_title('HAADF Reconstructed')
    ax[0].axis('off')

    #
    ax[1 + len(elem_names)].imshow((A.dot(xx ** gamma)).reshape(nx, ny)[135:235, 200:300], cmap='gray')
    ax[1 + len(elem_names)].set_title('HAADF Cropped Reconstructed')
    ax[1 + len(elem_names)].axis('off')

    #
    for ii in range(len(elem_names)):
        ax[ii + 1].imshow(xx[ii * (nx * ny):(ii + 1) * (nx * ny)].reshape(nx, ny), cmap='gray')
        ax[ii + 2 + len(elem_names)].imshow(xx[ii * (nx * ny):(ii + 1) * (nx * ny)].reshape(nx, ny)[135:235, 200:300],
                                            cmap='gray')

        ax[ii + 1].set_title(elem_names[ii] + ' Rec')
        ax[ii + 1].axis('off')
        ax[ii + 2 + len(elem_names)].set_title(elem_names[ii] + ' Cropped Rec')
        ax[ii + 2 + len(elem_names)].axis('off')

    fig.tight_layout()
    plt.show()