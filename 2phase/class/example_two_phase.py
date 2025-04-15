import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK
import time

ti.init(arch=ti.cpu)
import lbm_solver_3d_2phase_class as lb3dtp

time_init = time.time()
time_now = time.time()
time_pre = time.time()             
dt_count = 0 

lb3d = lb3dtp.LB3D_Solver_two_Phase(nx=131,ny=131,nz=131, sparse_storage=False)
             
solid_np, phase_np = lb3d.init_geo('./img_ftb131.txt','./phase_ftb131.dat')

#solid_np = init_geo('./img_ftb131.txt')
lb3d.solid.from_numpy(solid_np)
lb3d.psi.from_numpy(phase_np)
lb3d.init_simulation()

for iter in range(80000+1):
    lb3d.step()

    if (iter%500==0):
        
        time_pre = time_now
        time_now = time.time()
        diff_time = int(time_now-time_pre)
        elap_time = int(time_now-time_init)
        m_diff, s_diff = divmod(diff_time, 60)
        h_diff, m_diff = divmod(m_diff, 60)
        m_elap, s_elap = divmod(elap_time, 60)
        h_elap, m_elap = divmod(m_elap, 60)

        max_v = lb3d.get_max_v()
        
        print('----------Time between two outputs is %dh %dm %ds; elapsed time is %dh %dm %ds----------------------' %(h_diff, m_diff, s_diff,h_elap,m_elap,s_elap))
        print('The %dth iteration, Max Force = %f,  force_scale = %f\n\n ' %(iter, 10.0,  10.0))
        
        if (iter%10000==0):
            lb3d.export_VTK(iter)
            