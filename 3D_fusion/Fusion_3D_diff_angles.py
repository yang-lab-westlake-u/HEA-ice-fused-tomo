import Utils.mm_astra as mm_astra   
import Utils.utils_cs_eds as utils   
import Utils.pytvlib as pytvlib  
import matplotlib.pyplot as plt   
from tqdm import tqdm  
import numpy as np  
import scipy.io as sio   
import os
import h5py   


def display_recon_slices(slice_index, delta, save_path='recon_slices.png'):
    fig, ax = plt.subplots(1, len(elements), figsize=(20, 30))
    cmaps = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']   

    for i in range(len(elements)):
        start_slice = max(slice_index - int(delta / 2), 0)
        end_slice = min(slice_index + int(delta / 2), reconTotal.shape[2])

         
        slice_data = np.mean(reconTotal[i, :, start_slice:end_slice, :], axis=1)
         
        min_val = np.min(reconTotal[i])
        max_val = np.max(reconTotal[i])

         
        ax[i].imshow(slice_data, cmap=cmaps[i % len(cmaps)])   
        ax[i].set_title(f"{elements[i]}: min={min_val:.3f} max={max_val:.3f}")
        ax[i].axis('off')

     
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)   

  
haadf_data = sio.loadmat('./transposed_new_AuHEANW_241007t2_STEM_projs_align_241028.mat')
haadf = haadf_data['STEM_projs_new_n']   
 
if haadf.shape != (320, 320, 43):
    raise ValueError(f"HAADF data shape mismatch. Expected (320, 320, 43), but got {haadf.shape}.")
 
haadf[haadf < 0] = 0
 
element_files = ['transposed_projs_fused_Au.mat', 'transposed_projs_fused_Ag.mat', 'transposed_projs_fused_Bi.mat', 'transposed_projs_fused_Co.mat', 'transposed_projs_fused_Cu.mat', 'transposed_projs_fused_In.mat', 'transposed_projs_fused_Ni.mat', 'transposed_projs_fused_Pd.mat', 'transposed_projs_fused_Pt.mat', 'transposed_projs_fused_Rh.mat', 'transposed_projs_fused_Ru.mat']
elements = ['Au', 'Ag', 'Bi', 'Co', 'Cu', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Ru']
element_data = []
 
for i, element_file in enumerate(element_files):
    element_mat = sio.loadmat(f'./{element_file}') 
    dataset_name = f'fused_{elements[i]}' 
    if dataset_name not in element_mat:
        raise ValueError(f"Dataset {dataset_name} not found in {element_file}.")
    data = element_mat[dataset_name]
    if data.shape != (320, 320, 11):
        raise ValueError(f"Element {element_file} data shape mismatch. Expected (320, 320, 11), but got {data.shape}.")
    element_data.append(data)
for i in range(len(elements)):
    element_data[i][element_data[i] < 0] = 0
    #element_data[i] /= np.max(element_data[i])
    
haadf_tilt_angles = np.loadtxt('./tiltangles_HAADF.txt')  
NprojHAADF = haadf_tilt_angles.shape[0]
 
if haadf_tilt_angles.shape != (43,):
    raise ValueError(f"HAADF tilt angles shape mismatch. Expected (43,), but got {haadf_tilt_angles.shape}.")
 
chem_tilt_angles = np.loadtxt('./tiltangles_eds.txt')  
NprojCHEM = chem_tilt_angles.shape[0]
 
if chem_tilt_angles.shape != (11,):
    raise ValueError(f"EELS tilt angles shape mismatch. Expected (11,), but got {chem_tilt_angles.shape}.")

gamma = 1.6  
zNum = np.array([79, 47, 83, 27, 29, 49, 28, 46, 78, 45, 44], dtype=int)  
(nx, ny, _) = haadf.shape  
nPix = nx * ny  
nz = len(elements)  

fig, ax = plt.subplots(1, 11, figsize=(20, 20))  
ax = ax.flatten()

for i in range(len(elements)):
    ax[i].imshow(element_data[i][:, :, 5], cmap='gray')  
    ax[i].set_title(elements[i])
    ax[i].axis('off')
 
plt.savefig('1.png')

tomo = mm_astra.mm_astra(nx, ny, nz, np.deg2rad(haadf_tilt_angles), np.deg2rad(chem_tilt_angles))
 
bh = np.zeros([nx, ny * len(haadf_tilt_angles)])
for s in range(nx):
    bh[s, :] = haadf[s,].transpose().flatten()
tomo.set_haadf_tilt_series(bh)

bChem = np.zeros([nx, nx * len(chem_tilt_angles) * nz], dtype=element_data[0].dtype)
for ss in range(nx):
    bChem[ss, :] = np.concatenate([element_data[i][ss,].T.flatten() for i in range(nz)])
tomo.set_chem_tilt_series(bChem)
 
tomo.set_gamma(gamma)

sigmaMethod = 1
sigma = utils.create_weighted_summation_matrix(nx, nx, nz, zNum, 1.6, sigmaMethod)
(rows, cols) = sigma.nonzero()
vals = sigma.data
sigma = np.array([rows, cols, vals], dtype=np.float32, order='C')
tomo.load_sigma(sigma)

tomo.initialize_FP()
tomo.initialize_BP()
tomo.initialize_SIRT()
tomo.estimate_lipschitz() 
tomo.set_measureChem(True)
tomo.set_measureHaadf(True)
 
reconTotal = np.zeros([nz, nx, ny, ny], dtype=np.float32)

# Chemical Tomography with the Raw Data (Non-Fused Reconstructions)
tomo.restart_recon()
Niter = 250  # 
costCHEM = np.zeros(Niter)
for ii in tqdm(range(Niter)):
    costCHEM[ii] = tomo.poisson_ml(0.05)

plt.figure(figsize=(10, 3))
plt.plot(costCHEM)
plt.xlim([0, Niter - 1])
plt.xlabel('# Iterations')
plt.ylabel('Cost')
plt.savefig('poisson_ml_cost.png')
plt.close()
 
for e in range(nz):
    for s in range(nx):
        reconTotal[e, s,] = tomo.get_recon(e, s)
display_recon_slices(161, 2, '2.png')

#Now let's continue with the SIRT algorithm.
#tomo.restart_recon()

#Niter = 50 
#costCHEM = np.zeros(Niter)
#for i in tqdm(range(Niter)):
#    tomo.chemical_SIRT(5) 
#    costCHEM[i] = tomo.data_distance() 

#plt.figure(figsize=(10,3))
#plt.plot(costCHEM)
#plt.xlim([0,Niter-1])
#plt.xlabel('# Iterations')
#plt.ylabel('Cost')
#plt.savefig('sirt_cost.png')
#plt.close()

#for e in range(nz): 
#    for s in range(nx):
#        reconTotal[e,s,] = tomo.get_recon(e,s)
#display_recon_slices(172,2,'3.png')     #delta=2 


#Now let's wrap it up with the SART
#tomo.restart_recon()
#tomo.initialize_SART('sequential') 

#Niter = 15 
#costCHEM = np.zeros(Niter)
#for i in tqdm(range(Niter)):
#    tomo.chemical_SART(1)
#    costCHEM[i] = tomo.data_distance()

#plt.figure(figsize=(10, 3));
#plt.plot(costCHEM)
#plt.xlim([0, Niter - 1]);
#plt.xlabel('# Iterations');
#plt.ylabel('Cost')
#plt.savefig('sart_cost.png')
#plt.close()

#for e in range(nz): 
#    for s in range(nx):
#        reconTotal[e, s,] = tomo.get_recon(e, s)
#display_recon_slices(172, 2,'4.png')
 
utils.save_h5('Output_recon', 'raw_recon', **{elements[i]: reconTotal[i] for i in range(nz)})

# Now we can reload those preliminary states whenever we would like. 
def reload_starting_volume():
    scale = 10   

    reconTotal = np.zeros([nz, nx, ny, ny], dtype=np.float32)
    f = h5py.File('Output_recon/raw_recon.h5', 'r')
    for eInd, e in enumerate(elements):
        reconTotal[eInd,] = np.array(f[e])
    f.close()

    reconTotal *= scale    
    for e in range(nz):
        for s in range(nx):
            tomo.set_recon(reconTotal[e, s,], e, s)
 
    tomo.rescale_projections()
 
# Now that we have completed preprocessing and reconstructing the initial data,
# we can focus on solving the following cost function:

reload_starting_volume()   
 
reduceLambda = True   

Niter = 50 
tvIter = 5
iterSIRT = 5
lambdaTV = 0.005
lambdaCHEM = 0.05
lambdaHAADF = 0.09
 
costCHEM = np.zeros(Niter, dtype=np.float32)
costHAADF = costCHEM.copy()
costTV = costCHEM.copy()
params = {'lambdaTV': lambdaTV, 'tvIter': tvIter, 'Niter': Niter, 'gamma': gamma,
          'lambdaCHEM': lambdaCHEM, 'lambdaHAADF': lambdaHAADF, 'iterSIRT': iterSIRT,
          'sigmaMethod': sigmaMethod, 'reduceLambda': reduceLambda}


# 1. 
def save_recon_data(tomo, file_path):
    
    recon_data = np.zeros([nz, nx, ny, ny], dtype=np.float32)

    for e in range(nz):
        for s in range(nx):
            recon_data[e, s,] = tomo.get_recon(e, s) 
    with h5py.File(file_path, 'w') as f:
        for i, element in enumerate(elements):
            f.create_dataset(element, data=recon_data[i], compression="gzip")


# 2. 
recon_save_path = 'Output_recon/recon_data_after_fusion.h5'
 
for i in tqdm(range(Niter)):
    costHAADF[i], costCHEM[i] = tomo.sirt_data_fusion(lambdaHAADF, lambdaCHEM, iterSIRT)   
    costTV[i] = tomo.tv_fgp_4D(tvIter, lambdaTV)   
    if i > 0 and costHAADF[i] > costHAADF[i - 1]:   
        lambdaCHEM *= 0.95
 
save_recon_data(tomo, recon_save_path)
 
for e in range(nz):
    for s in range(nx):
        reconTotal[e, s,] = tomo.get_recon(e, s)
 
save_dir = 'Output_recon'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'fusion_recon.h5')

with h5py.File(save_path, 'w') as f:
    for i, element in enumerate(elements):
        # 
        f.create_dataset(element, data=reconTotal[i], compression="gzip")
        print(f"Saved fusion reconstruction for {element} to {save_path}")
 
plt.figure(figsize=(25, 3))

ax1 = plt.subplot(1, 3, 1)
ax1.set_ylabel(r'$||A (\Sigma x) - b||^2$')
ax2 = plt.subplot(1, 3, 2)
ax2.set_ylabel(r'$\sum (Ax - b \cdot \log(Ax))$')
ax3 = plt.subplot(1, 3, 3)
ax3.set_ylabel(r'$\sum \|x\|_{TV}$')
ax1.plot(costHAADF)
ax2.plot(costCHEM)
ax3.plot(costTV)
plt.savefig('data_fusion_cost.png')
plt.close()
 
display_recon_slices(161, 2, '5.png')
 
g = tomo.get_model_projections()   
bhTemp = tomo.get_haadf_projections()   
 
def reshape_300x3300_to_300x300x43(array_320x3520):
        reshaped_array = np.zeros((320, 320, 43), dtype=array_320x3520.dtype)

    for i in range(43):
        reshaped_array[:, :, i] = array_320x3520[:, i * 320:(i + 1) * 320]   

    return reshaped_array
 
g_reshaped = reshape_300x3300_to_300x300x43(g)
bhTemp_reshaped = reshape_300x3300_to_300x300x43(bhTemp)
 
save_dir = 'Output_recon'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'model_and_haadf_projections.h5')

with h5py.File(save_path, 'w') as f: 
    f.create_dataset('model_projections', data=g_reshaped, compression="gzip")
    print(f"Saved reshaped model projections to {save_path}")

    f.create_dataset('haadf_projections', data=bhTemp_reshaped, compression="gzip")
    print(f"Saved reshaped HAADF projections to {save_path}")
 
ind = 22   

plt.figure(figsize=(13,13))
ax1 = plt.subplot(1,2,1)
ax1.imshow(bhTemp[:,ny*ind:ny*(ind+1)],cmap='gray')
ax1.set_title('Measured Projection: '+str(np.round(np.max(bhTemp[:,:320]),2))) 


ax2 = plt.subplot(1,2,2)
ax2.imshow(g[:,ny*ind:ny*(ind+1)],cmap='gray') #YlGnBu
ax2.set_title('Reprojection: '+str(np.round(np.max(g[:,:320]),2)))
plt.savefig('projection_comparison.png')
plt.close()
