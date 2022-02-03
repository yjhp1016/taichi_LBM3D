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
 - name: Department of Chemical Engineering, Imperial College London, UK
   index: 1
 - name: Current address TotalEnergies E&P UK Limited
   index: 2
 - name: Centre for Renewable Energy Systems, Department of Energy and Power, Cranfield University, MK43 0AL, UK
   index: 3
 
date: 12 September 2021
bibliography: paper.bib
---

# Summary
Taichi_LBM3D is an open-source single and two-phase 3D lattice Boltzmann solver with D3Q19 lattice model [@qian1992lattice], Multi-Relaxation-Time collision scheme [@d1992generalized] and sparse storage structure implemented using Taichi programming language [@hu2020taichi], which is designed for single and two-phase porous medium flow simulation. The fluid density on a lattice is simulated with a streaming and collision process. Taking advantage of Taichi's computing structure, Taichi_LBM3D can be employed on shared-memory multi-core CPUs or massively parallel GPUs (OpenGL and CUDA). The code is around 400 lines, extensible and intuitive to understand. This solver is especially efficient for study of complex flow in porous medium [@yang2019image; @yang2013quantitative].

# Statement of Need
In the past two decades, several well developed code packages emerged, including Palabos, OpenLB, Walbera and many others can be found on Github. However, they are mostly implemented in C++ for performance reasons. Understanding and modifying these packages will take researchers a lot of effort and extra time, especially the parallel computing parts are usually not straightforward to learn and implement. These packages can be run only on CPU structure. In order to maintain a high computing efficiency, the researchers need to spend significantly on low-level coding engineering which distracts their attention from high-level thinking of physics and algorithms. We developed a 3D single phase MRT LBM solver and a 3D two-phase improved color gradient MRT solver [@ahrenholz2008prediction; @tolke2006adaptive] using Taichi programming languages, which is a novel high performance computing language [@hu2020taichi]. The objective of this new LBM implementation is to facilitate researchers to focus on the LB algorithm or application but not coding side, while still guaranteeing the high efficiency of the computation on various computing structures (CPUs and GPUs). The researchers can rapidly prototyping their new algorithm and/or test their new applications with excellent efficiency. 

Based on this excellent computing infrastructure, these two solvers (full functional LBM code with pressure/velocity boundary conditions and force term) can be execute parallel in shared memory system in CPU backend (e.g. x64, ARM64) and GPUs (CUDA, Metal and OpenGL). The source code is very short: 400 lines for single phase LBM and 500 lines for two-phase solver. All the modules are implemented using python-like syntax along with Taichi embedded vector/matrix operations and are very intuitive to understand. For example the collision operation module was implemented with only ~30 lines of code. This unique feature makes the further extension straightforward for researchers and keeps the coding cost minimum in order to free their time on the physics and algorithm side. These solvers can be run in sparse storage mode which is an essential feature for flow in porous medium simulation, this memory operation is decoupled from the computing side, only several extra lines to define desired sparse storage structure without any change on the code of the computing kernel. We will also show some test cases that the computing performance is excellent compared to other C++ implementations and even more attractive if GPUs are used as computing backend.

Researchers in computing physics can use this code package to rapidly test their new ideas and new applications on various computing backends (CPUs and GPUs) without losing any computing efficiency, while at the same time the cost of code implementation can be potentially reduced from days to roughly several hours.

# Algorithm
Several lattice models have been proposed for the LB method. The most popular and widely used lattice model has been used in 2D and 3D was proposed called D3Q19 [@qian1992lattice]. The model contains 19 velocities at each lattice node. We use this lattice model as our LB solver implementation. The collision term is the key component in the LB method, it defines how particle groups exchange momentum and energy locally at lattice nodes. The simplest one that can be used for flow simulation is the Bhatnagar-Gross-Krook (BGK) operator. To overcome the disadvantages of the BGK model, such as numerical instabilities, multiple-relaxation-time (MRT) scheme was proposed [@d2002multiple]. The main idea of MRT is using different relaxation time parameters for different moments of macroscopic quantities. The stability improvement by using the MRT scheme would reduce the computational effort by at least one order of magnitude while maintaining the accuracy of the simulations. We use the MRT scheme as a collision term in our single and two-phase solver development.

Due to the meso-scopic nature of the LB method, it is very suitable to extend to multiphase, multiphysics application. There have been many multiphase/multicomponent LB models proposed in the past two decades. These models generally can be grouped into four categories: Colour gradient model [@gunstensen1991lattice; @grunau1993lattice; @lishchuk2003lattice; @ahrenholz2008prediction], the free energy model [@swift1995lattice;@inamuro2000galilean], pseudopotential Shane-Chen model [@shan1993lattice;@shan1994simulation;@sbragaglia2007generalized] and phase-field model [@liu2013phase]. We implement an optimized colour-gradient model  proposed in [@ahrenholz2008prediction] which permits improved numerical stability, higher viscosity ratio and lower capillary number compared to other two-phase models. 

# Acknowledgement
We acknowledge the support from Taichi's developing team.

# References


