import time
import taichi as ti

ti.init(arch=ti.cpu, dynamic_index=False, kernel_profiler=False, print_ir=False)
import LBM_3D_SinglePhase_Solver as lb3dsp

time_init = time.time()
time_now = time.time()
time_pre = time.time()             


lb3d = lb3dsp.LB3D_Solver_Single_Phase(nx=50,ny=50,nz=50, sparse_storage=False)

lb3d.init_geo('./geo_cavity.dat')
lb3d.set_bc_vel_x1([0.0,0.0,0.1])
lb3d.init_simulation()

for iter in range(2000+1):
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
        print('The %dth iteration, Max Force = %f,  force_scale = %f\n\n ' %(iter, max_v,  0.0))
        
        if (iter%1000==0):
            lb3d.export_VTK(iter)
            
