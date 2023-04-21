lbm_solver_3d_Macro_Sukop
=================================

This solver is almost similar to lbm_solver_3d expect several difference as follows:

1. Some parameter is different

.. code-block:: python

    #grid resolution
    nx,ny,nz = 60,50,5
    #external force
    fx,fy,fz = 1.0e-6,0.0,0.0
    #viscosity
    niu = 0.1
    #import geometry
    geo_name = './BC.dat'
    #maximum timestep
    max_timestep = 5000
    #output frequency
    output_fre = 100
    #vtk file output frequency
    vtk_fre = 500

2. There are two array for solid flag data.

.. code-block:: python

    ns_np = init_geo(geo_name)
    solid_np = ns_np.astype(int) 
    #solid_np = init_geo('./img_ftb131.txt')
    solid.from_numpy(solid_np)
    ns.from_numpy(ns_np)

3. The streaming function is different

.. code-block:: python

    @ti.kernel
    def streaming0():
        for i in ti.grouped(rho):
            if (solid[i] == 0):
                for s in ti.static(range(19)):
                    ip = periodic_index(i+e[s])
                    #if it is fluid f2=f otherwise apply bounce-back f2[i,s]=f[ip,LR[s]]
                    f2[i,s] = f[i,s] + ns[i]*(f[ip,LR[s]] - f[i,s])


    @ti.kernel
    def streaming1():
        for i in ti.grouped(rho):
            if (solid[i] == 0):
                #if it is fluid apply streaming 
                for s in ti.static(range(19)):
                    ip = periodic_index(i+e[s])
                    F[ip,s] = f2[i,s]
                    
                    #if (solid[ip]==0):
                    #    F[ip,s] = f[i,s]
                    #else:
                    #    F[i,LR[s]] = f[i,s]
                        #print(i, ip, "@@@")
    #not used
    @ti.kernel
    def streaming2():
        for i in ti.grouped(rho):
            for s in ti.static(range(19)):
                f[i,s] = F[i,s]

``streaming3()`` calculates the macroscopic variable