---
title: 'Taichi_LBM3D: A 3D lattice Bolzmann solver in Taichi'
tags:
  - Taichi
  - Digital Rock Physics
  - lattice Boltzmann method
  - Computational fluid dynamics
authors:
  - name: Jianhui Yang
    orcid: 0000-0001-9622-2713
    affiliation: "1,2"
  - name: Liang Yang^[Corresponding author]
    orcid: 0000-0003-0901-0929
    affiliation: 3
affiliations:
 - name: Department of Chemical Engineering, Imperial College London
   index: 1
 - name: Current address TotalEnergies E&P UK Limited
   index : 2
 - name: Cranfield University
   index: 3
  
date: 12 September 2021
bibliography: paper.bib
---

# Summary
Taichi_LBM3D is an open-source single and two-phase 3D lattice Boltzmann solver with D3Q19 lattice model [@qian1992lattice], Multi-Relaxation-Time collision scheme [@d1992generalized] and sparse storage structure implemented using Taichi programming language[@hu2020taichi], which is designed for single and two-phase porous medium flow simulation. The fluid density on a lattice is simulated with streaming and collision process. Taking advantage of Taichi's computing structure, Taichi_LBM3D can be employed on shared-memory multi-core CPUs or massively parallel GPUs (OpenGL and CUDA). The code is around 400 lines, extensible and intuitive to understand. This solver is especially efficient for study of complex flow in porous medium[@yang2019image; @yang2013quantitative].

# Statement of Need
In the past two decades, several well developed code packages emerged, including Palabos, OpenLB, Walbera and may others can be found on Github. However, they are mostly implemented in C++ for performance reasons. Understanding and modifying these packages will take researchers a lot of effort and extra time, especially the parallel computing parts are usually not straightforward to learn and implement. These packages can be run only on CPU structure. In order to maintain a high computing efficiency, the researchers need to spend significant on low-level coding engineering which distract their attention from high-level thinking of physics and algorithms. We developed a 3D single phase MRT LBM solver and a 3D two-phase improved color gradient MRT solver [@ahrenholz2008prediction; @tolke2006adaptive] using Taichi programming languages, which is a novel high performance computing language [@hu2020taichi]. The objective of this new LBM implementation is to facilitate researchers to focus on the LB algorithm or application but not coding side, while still guarantee the high efficiency of the computation on various computing structures (CPUs and GPUs). The researchers can rapidly prototyping their new algorithm and/or test their new applications with excellent efficiency. 

Based on this excellent computing infrastructure, these two solvers (full functional LBM code with pressure/velocity boundary conditions and force term) can be execute parallel in shared memory system in CPU backend (e.g. x64, ARM64) and GPUs (CUDA, Metal and OpenGL). The source code is very short: 400 lines for single phase LBM and 500 lines for two-phase solver. All the modules are implemented using python-like syntax along with Taichi embedded vector/matrix operation are very intuitive to understand. For example the collision operation module were implemented with only ~30 lines of code. This unique feature makes the further extension straightforward for researchers and keep the coding cost minimum in order to free their time on physics and algorithm side. These solvers can be run in sparse storage mode which is an essential feature for flow in porous medium simulation, this memory operation is decoupled from computing side, only several extra lines to define desired sparse storage structure without any change on the code of computing kernel. We will also show some test case that the computing performance is excellent compared to other C++ implementations and even more attractive if GPUs is used as computing backend.

Researchers in computing physics can use this code package to rapidly test their new ideas and new applications on various computing backends (CPUs and GPUs) without losing any computing efficiency, while at the same time the cost of code implementation can be potentially reduced from days to roughly several hours.


# Acknowledgement

# References


