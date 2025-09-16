# Synthesising high-entropy alloy nanoparticles, aerogels and coatings using a bilayer ice recrystallization method

## Contents

1. [Overview](#1-overview)  
2. [System Requirements](#2-system-requirements)  
3. [Installation Guide](#3-installation-guide)  
4. [Source Codes](#4-source-codes)
5. [Learn More](#5-learn-more)


---

## 1. Overview
High-entropy alloys (HEAs) are usually synthesized by stabilizing thermodynamically metastable structures from high temperatures. Here, we present a bilayer ice recrystallization (BLIR) approach performed entirely at subzero temperatures to synthesize HEA nanoparticles (HEA-NPs) or aerogels (HEAAs) with up to 11 metal elements. The experimental data, instruction and data analysis source codes for fused multi-modal electron tomography are provided here.

## 2. System Requirements

### 2.1 Operating System

Linux CentOS 7.9.2009.

### 2.2 Python version

This package has been tested with Python 3.12. 
We recommend the use of Python version 3.12 or higher to test the data and source codes.

It also required following packages in Python:

(1) gcc/9.3;     
(2) cuda/11.8;     
(3) boost/1.8.5;     
(4) openmpi-4.1.1;    
(5) eigen-3.4.0  
(6) astra-toolbox(https://github.com/jtschwar/tomo_TV/tree/master/gpu_3D/Utils)  
(7) tomo_TV(https://github.com/jtschwar/tomo_TV)

### 2.3 Matlab version

This package has been tested with Matlab R2024a.  
We recommend the use of Matlab version R2021a or higher to test the data and source codes.

## 3. Installation Guide

We use the supercomputer's GPU for computation (256G DRAM, 16-core CPU and 1 GPU). Typical install time is approximately 5 mins.  
The pipelines can be found from the link: https://github.com/jtschwar/tomo_TV from the following reference:  
[1] J. Schwartz, Z.W. Di, Y. Jiang, et al. "Imaging 3D Chemistry at 1 nm Resolution with Fused Multi-Modal Electron Tomography", Nature Communications, 15(1), 3555 (2024).

## 4. Source Codes

### 4.1 Demonstration codes  
Folder: [_demo](./_demo)

Run source data/_demo/Fusion_3D_diff_angles_CoMnO_mat.py to test the demonstration sample of fused tomography data on CoMnO nanoparticles.

### 4.2 HEA coated nanowire codes
Folder: [3D_fusion](./3D_fusion)

### 4.3 reconstructed 3D chemical distributions  
Run source data/3D_fusion/input/Fusion_3D_diff_angles.py to generate the output file fusion_recon.h5, which contains all the reconstructed 3D chemical distributions.  

## 5. Learn More
For more information about the source data, please visit:

- [Yanglab@Westlake_University](https://em.lab.westlake.edu.cn/info/1006/1293.htm)
