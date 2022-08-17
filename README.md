# taichi_LBM3D
## Background
Taichi_LBM3D is a 3D lattice Boltzmann solver with Multi-Relaxation-Time collision scheme and sparse storage structure implemented using [Taichi programming language](https://github.com/taichi-dev/taichi), which is designed for porous medium flow simulation. Taking advantage of Taichi's computing structure, Taichi_LBM3D can be employed on shared-memory multi-core CPUs or massively parallel GPUs (OpenGL and CUDA). **The code achieved 900 MLUPS in a 3D lid-driven test case using a A100 NVIDIA GPU**. The code is only around 400 lines, highly extensible and very intuitive to understand.

## Installation
This solver is developed using Taichi programming language (a python embedded programming language), install [Taichi](https://github.com/taichi-dev/taichi) is required, by `python3 -m pip install taichi`.  

Pyevtk is required for export simualtion result for visualization in Paraview, install [Pyevtk](https://pypi.org/project/pyevtk/) by `pip install pyevtk`

## Usage
There are several place for users to modify to fit their problems:
###### set computing backend
First the computing backend should be specified by `ti.init(arch=ti.cpu)` *using parallel CPU backend*, or by `ti.init(arch=ti.gpu)` *to use OpenGL or CUDA(is available) as computing backend*
###### set input geometry
LBM uses uniform mesh, the geometry is import as a ASCII file with 0 and 1, where 0 represent fluid point and 1 represent solid point. They are stored in format:
```
for k in range(nz)
  for j in range(ny)
    for i in range(nx)
      geometry[i,j,k]
```
You can specify the input file at:
`solid_np = init_geo('./img_ftb131.txt')`

For two phase solver, a two phase distribution input file is also requred. This file is composed of -1 and 1 representing phase 1 and 2 respectively

###### set geometry size
Set geometry input file size here: `nx,ny,nz = 131,131,131`

###### set external force
Set expernal force applied on the fluid here: `fx,fy,fz = 0.0e-6,0.0,0.0`

###### set boundary conditions
There are three boundary conditions used in this code: Periodic boundary condition, fix pressure boundary condition, and fix velocity boundary condition
We use the left side of X direction as an example: `bc_x_left, rho_bcxl, vx_bcxl, vy_bcxl, vz_bcxl = 1, 1.0, 0.0e-5, 0.0, 0.0`
set boundary condition type in `bc_x_left`; 0=periodic boundary condition, 1 = fix pressure boundary condition, 2 = fix velocity boundary condition
if `bc_x_left == 1` is select, then the desired pressure on the left side of X direction need to be given in `rho_bcxl`
if `bc_x_left == 2` is select, then the desired velocity on the left side of X direction need to be given in `vx_bcxl, vy_bcxl, vz_bcxl`

The same rules applied to the other five sides

###### set viscosity
Viscosity is set in `niu = 0.1` for single phase solver
```
niu_l = 0.05
niu_g = 0.2
```
for two phase solver, niu_l for liquid phase, niu_g for phase 2

###### Additional parameters for two phase solver
- Contact angle of the solid surface can be specified in `psi_solid = 0.7` this value is the cosine of the desired contact angle, so the value is between -1 and 1
- Interfical tension of two phases is set in `CapA = 0.005`
- Boundary condition for the phase setting: `bc_psi_x_left, psi_x_left = 1, -1.0 ` bc_psi_x_left = 0 for periodic boundary for the phase field, 1 = constant phase field value boundary. If bc_psi_x_left is set as 1, then the next parameter is desired constant phase for this boundary: psi_x_left should be set as -1.0 or 1.0 for phase 1 or phase 2 respectively. 


**All the quantities are in lattice units**



## Examples (Direct Numerical Simulation)

###### Flow over a vehicle: inertia dominated
![image](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/car1.png)
![image](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/car2.png)

###### Single phase flow in a sandstone (Sandstone geometry is build from Micro-CT images at 7.5 microns): viscous dominated
![image](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/ftb1.png)

###### Urban air flow: inertia dominated
![image](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/city1.png)

###### Two Phase flow: oil (non-wetting phase) into a ketton carbonate rock saturated with water (wetting phase): capillary dominated
![Alt text](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/ket_drain.gif)

## Authors
Jianhui Yang @yjhp1016
Liang Yang @ly16302

## License 
MIT

## Citations
If you use Taichi-LBM3D in your research, please cite the corresponding paper:

    @Article{fluids7080270,
     AUTHOR = {Yang, Jianhui and Xu, Yi and Yang, Liang},
     TITLE = {Taichi-LBM3D: A Single-Phase and Multiphase Lattice Boltzmann Solver on Cross-Platform Multicore CPU/GPUs},
     JOURNAL = {Fluids},
     VOLUME = {7},
     YEAR = {2022},
     NUMBER = {8},
     ARTICLE-NUMBER = {270},
     URL = {https://www.mdpi.com/2311-5521/7/8/270},
     ISSN = {2311-5521},
     DOI = {10.3390/fluids7080270}
    }
    
### To do list:
- wrap functions into class to faciliate direct use from users
- Modify single phase gray scale and two phase data structure to improve computing efficiency (single phase code is done)
- Packing through pip
