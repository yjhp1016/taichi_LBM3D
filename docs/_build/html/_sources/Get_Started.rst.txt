Get_Started
=====================

Introduction
-----------------

This solver has four seperate solver which are :doc:`Single_phase`, :doc:`Grey_Scale`, :doc:`2phase` and :doc:`Phase_change`
The :doc:`Single_phase` and :doc:`Grey_Scale` are single phase LBM solver based on D3Q19 MRT scheme which are used for simulating the flow in a single phase medium.
The :doc:`2phase` and :doc:`Phase_change` are two phase color gradient LBM solver which are used for simulating the flow in a two phase medium.

In the :doc:`Single_phase` folder there are one solver file with objected oriented programming and two example file for using this solver. 
There are also three files which do not use objected oriented programming but combine the solver and the simulation together.

In the :doc:`Grey_Scale` folder there are one file combined solver and case, and another file for input file generation.

In the :doc:`2phase` folder there are two solver files with combined multiphase solver which is based on color gradient method and case.

In the :doc:`Phase_change` folder there are one solver files with objected oriented programming which inherits the Single_phase solver and add solute solver and color gradient multiphase solver. 

Usage
-------------

To use this solver, you need to install the following packages:: 

    python3 -m pip install taichi
    pip install pyevtk

And then you can run the solver by::

    python3 example_file_name.py

The example file can be found in :doc:`Single_phase` and :doc:`Phase_change` folder. 
However, in the :doc:`2phase` folder, the solver and the case are combined in one file, so you can run the simulation directly by::

    python3 lbm_solver_3d_Macro_Sukop.py 
    python3 lbm_solver_3d_2phase.py

In terms of the :doc:`Grey_Scale` solver, you need to run the input file generation file first and then run the solver file::
    
    python3 flow_domain_geo_generation.py
    python3 lbm_solver_3d_Macro_Sukop.py

Write an case file 
---------------------------

As can be seen from the example file in the :doc:`Single_phase`. To use the solver, you need to first import taichi,time package and the solver. Then you need to initialize taichi package
by using the following code::

    ti.init(arch=ti.cpu, dynamic_index=False, kernel_profiler=False, print_ir=False)

The next step is to create instance of the solver by using the following code::

    lb3d = lb3dsp.LB3D_Solver_Single_Phase(nx=50,ny=50,nz=50, sparse_storage=False)

where you can set the size of the domain by changing the nx,ny,nz. The sparse_storage is used to determine whether to use sparse storage or not.

Then you have to read the initial geometry file, set the boundary condition and initialize the solver by using the following code::

    lb3d.init_geo('./geo_cavity.dat')
    lb3d.set_bc_vel_x1([0.0,0.0,0.1])
    lb3d.init_simulation()

Then you can run the simulation by using the following code::

    for iter in range(2000+1):
        lb3d.step()

Finally, you can record the time ,print some information and output the results to VTK file by using the following code::

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

Here you can set the frequency of output by changing the number in the if statement. You can also change the frequency of printing the information by changing the number in the if statement.





