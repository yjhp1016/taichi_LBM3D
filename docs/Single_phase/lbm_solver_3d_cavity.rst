lbm_solver_3d_cavity
=================================

This solver is almost similar to lbm_solver_3d expect several difference as follows:

1. The Grid resolution in this solver is 50x50x50
2. The viscosity in this solver is 0.16667
3. The boundary condition in this solver is velocity solver on x-right as follows

boundary condition of this solver

.. code-block:: python

    #Boundary condition mode: 0=periodic, 1= fix pressure, 2=fix velocity; boundary pressure value (rho); boundary velocity value for vx,vy,vz
    bc_x_left, rho_bcxl, vx_bcxl, vy_bcxl, vz_bcxl = 0, 1.0, 0.0e-5, 0.0, 0.0  #Boundary x-axis left side
    bc_x_right, rho_bcxr, vx_bcxr, vy_bcxr, vz_bcxr = 2, 1.0, 0.0, 0.0, 0.1  #Boundary x-axis right side
    bc_y_left, rho_bcyl, vx_bcyl, vy_bcyl, vz_bcyl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary y-axis left side
    bc_y_right, rho_bcyr, vx_bcyr, vy_bcyr, vz_bcyr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary y-axis right side
    bc_z_left, rho_bczl, vx_bczl, vy_bczl, vz_bczl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary z-axis left side
    bc_z_right, rho_bczr, vx_bczr, vy_bczr, vz_bczr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary z-axis right side

x-right is implementated with velocity boundary condition

4. The boundary condition implementation is different from lbm_solver_3d, in this solver, the density distribution
function is calculated based on velocity on the boundary.

.. code-block:: python

    if ti.static(bc_x_left==2):
            for j,k in ti.ndrange((0,ny),(0,nz)):
                if (solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        #F[0,j,k][s]=feq(LR[s], 1.0, bc_vel_x_left[None])-F[0,j,k,LR[s]]+feq(s,1.0,bc_vel_x_left[None])  #!!!!!!change velocity in feq into vector
                        F[0,j,k][s]=feq(s,1.0,ti.Vector(bc_vel_x_left))


5. Finally, the definition of the varible is slightly different from lbm_solver_3d