# taichi_LBM3D
## Background
Taichi_LBM3D is a 3D lattice Boltzmann solver with Multi-Relaxation-Time collision scheme and sparse storage structure implemented using [Taichi programming language](https://github.com/taichi-dev/taichi), which is designed for porous medium flow simulation. Taking advantage of Taichi's computing structure, Taichi_LBM3D can be employed on shared-memory multi-core CPUs or massively parallel GPUs (OpenGL and CUDA). The code is around 400 lines, extensible and intuitive to understand.

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

The same rule applied to the other five sides

###### set viscosity
Viscosity is set in `niu = 0.1`

All the quantities are in lattice units

## Examples (Direct Numerical Simulation)

###### Flow over a vehicle: inertia dominated
![image](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/car1.png)
![image](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/car2.png)

###### Single phase flow in a sandstone (Sandstone geometry is build from Micro-CT images at 7.5 microns): viscous dominated
![image](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/ftb1.png)

###### Urban air flow: inertia dominated
![image](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/city1.png)

###### Two Phase flow: oil (non-wetting phase) into a ketton carbonate rock saturated with water (wetting phase)
![Alt text](https://github.com/yjhp1016/taichi_LBM3D/blob/main/img/ket_drain.gif)

## Authors
Jianhui Yang @yjhp1016
Liang Yang @ly16302

## License 
MIT
