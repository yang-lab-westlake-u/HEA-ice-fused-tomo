Making high-entropy alloy nanoparticles, aerogels and coatings by ice

The instruction for fused multi-modal electron tomography.

1. System requirements:

1.1 This package has been tested on the following Operating System: Linux CentOS 7.9.2009.

1.2 Python version requirements: This package has been tested with Python 3.12. 
We recommend the use of Python version 3.12 or higher to test the data and source codes.
It also required following packages in Python:
(1) gcc/9.3;     (2) cuda/11.8;     (3) boost/1.8.5;     (4) openmpi-4.1.1;     (5) eigen-3.4.0
(6) astra-toolbox(https://github.com/jtschwar/tomo_TV/tree/master/gpu_3D/Utils)
(7) tomo_TV(https://github.com/jtschwar/tomo_TV)

1.3 Matlab version requirements: This package has been tested with Matlab R2024a. 
We recommend the use of Matlab version R2021a or higher to test the data and source codes.

2. Installation guide:

We use the supercomputer's GPU for computation (256G DRAM, 16-core CPU and 1 GPU). Typical install time is approximately 5 mins.
The pipelines can be found from the link: https://github.com/jtschwar/tomo_TV from the following reference:
[1] J. Schwartz, Z.W. Di, Y. Jiang, et al. "Imaging 3D Chemistry at 1 nm Resolution with Fused Multi-Modal Electron Tomography", Nature Communications, 15(1), 3555 (2024).

3. Demonstration of a sample data

Run source data/_demo/Fusion_3D_diff_angles_CoMnO_mat.py to test the demonstration sample of fused tomography data on CoMnO nanoparticles.

4. Instructions for reproduction of fused multi-modal electron tomography of HEA coated nanowire

4.1 Run source data/2D_fusion/Au_NW_new.py to obtain the reconstructed 2D chemical distributions at different projection angles.

4.2 Run source data/2D_fusion/output/h5_merge.py to merge the projections of the same element at different angles.

4.3 Run source data/3D_fusion/input/transposed.m to transpose and merge projections obtained from 4.2 for subsequent 3D multi-modal reconstruction and visualization.

4.4 Run source data/3D_fusion/input/Fusion_3D_diff_angles.py to generate the output file fusion_recon.h5, which contains all the reconstructed 3D chemical distributions.

4.5 Run source data/3D_fusion/reconstruction_volumes/step1_Chem_total.m and 3D_fusion/reconstruction_volumes/step2_Mask.m files to apply a mask to the reconstructed 3D chemical distributions in fusion_recon.h5 and obtain the final chemical tomograms for different elements.
