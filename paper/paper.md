---
title: 'Taichi_LBM3D: A 3D lattice Bolzmann solver in Taichi'
tags:
  - Taichi
  - Digital Rock Physics
  - lattice Boltzmann method
  - Computational fluid dynamics
authors:
  - name: Jianhui Yang ^[corresponding author]
    orcid: 0000-0001-9622-2713
    affiliation: 1
  - name: Liang Yang
    orcid: 0000-0003-0901-0929
    affiliation: 2
affiliations:
 - name: 
   index: 1
 - name: Cranfield University
   index: 2
  
date: 12 September 2021
bibliography: paper.bib
---

# Summary
Taichi_LBM3D is an open-source 3D lattice Boltzmann solver with Multi-Relaxation-Time collision scheme and sparse storage structure implemented using Taichi programming language, which is designed for porous medium flow simulation. Taking advantage of Taichi's computing structure, Taichi_LBM3D can be employed on shared-memory multi-core CPUs or massively parallel GPUs (OpenGL and CUDA). The code is around 400 lines, extensible and intuitive to understand.

# Statement of Need
Understanding the flow over porous medium and calculation the flow field is important in petroleum engineering, earth science and enviromental engineering problem. With the advancement of micro-CT imaging with a very fine resolition, it can provide the pore space of many reservoir rocks. So it is possible to solve the flow equation directly within porous rock. There are several approaches, including finite volume method (FVM) [@bijeljic2013predictions], finite element method (FEM) [@yang2019image]and lattice Boltzmann method (LBM) [@yang2013comparison]. LBM is a simplified Boltzmann transport solver in lattices, with a relaxation time. However, there are a few LBM code which can run efficiently on GPUs, especially with a sparse storage structure. 

Taichi_LBM3D was developed for academic and industrial researchers in the field of Digital Rock Physics, but it can also simulate the inertia dominated flow, for example, flow over vehicles, urban air flows, etc. 

# References

# Acknowledgement
We acknowledge the support from Taichi's developer team. 

