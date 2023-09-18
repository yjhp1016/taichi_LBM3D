import time
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, kernel_profiler=False, print_ir=False)
import LBM_3D_SinglePhase_Solver as lb3dsp

time_init = time.time()
time_now = time.time()
time_pre = time.time()             

NX = 5
NY = 20
NZ = 16

lb3d = lb3dsp.LB3D_Solver_Single_Phase(nx=NX,ny=NY,nz=NZ, sparse_storage=False)

geometry = np.zeros((NX, NY, NZ))
geometry[:,:,0] = 1
geometry[:,:,-1] = 1


#lb3d.init_geo('./geo_cavity.dat')
lb3d.solid.from_numpy(geometry)

lb3d.set_force([0.0,0.0001,0.0])
lb3d.set_viscosity(0.1667)

lb3d.init_simulation()

for iter in range(150000+1):
    lb3d.step()

    if (iter%2000==0):
        
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
        print('The %dth iteration, Max Force = %f,  force_scale = %f\n\n ' %(iter, max_v,  lb3d.fz))
        
        
        if (iter%1000==0):
            lb3d.export_VTK(iter)
            
