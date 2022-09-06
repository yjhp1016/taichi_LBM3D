import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK
import time

ti.init(arch=ti.gpu, dynamic_index=False, kernel_profiler=False, print_ir=False)
import LBM_3D_SinglePhase_Solute_Solver as lb3d_solute_solver

time_init = time.time()
time_now = time.time()
time_pre = time.time()
dt_count = 0            


lb3d_solute = lb3d_solute_solver.LB3D_Solver_Single_Phase_Solute(50,50,5)
lb3d_solute.init_geo('./geo_cavity.dat')
lb3d_solute.init_concentration('./psi.dat')

lb3d_solute.set_force([0e-6,-1.0e-6,0.0])
lb3d_solute.set_viscosity(0.05)
lb3d_solute.init_solute_simulation()


for iter in range(300000+1):
    lb3d_solute.step()

    if (iter%1000==0):
        
        time_pre = time_now
        time_now = time.time()
        diff_time = int(time_now-time_pre)
        elap_time = int(time_now-time_init)
        m_diff, s_diff = divmod(diff_time, 60)
        h_diff, m_diff = divmod(m_diff, 60)
        m_elap, s_elap = divmod(elap_time, 60)
        h_elap, m_elap = divmod(m_elap, 60)

        max_v = lb3d_solute.get_max_v()
        
        print('----------Time between two outputs is %dh %dm %ds; elapsed time is %dh %dm %ds----------------------' %(h_diff, m_diff, s_diff,h_elap,m_elap,s_elap))
        print('The %dth iteration, Max Force = %f,  force_scale = %f\n\n ' %(iter, max_v,  10.0))
        
        if (iter%5000==0):
            lb3d_solute.export_VTK(iter)
