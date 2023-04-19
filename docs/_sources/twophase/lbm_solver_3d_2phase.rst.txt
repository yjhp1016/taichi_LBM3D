lbm_solver_3d_2phase
========================================

This solver is the multiphase model based on color gradient model 
Firstly, it defines some parameters

.. code-block:: python 

    # NOTE: THIS CODE NEED taichi_glsl, so please use taichi version <=0.8.5
    #import taichi, numpy, pyevtk and time package
    import taichi as ti
    import numpy as np
    #import taichi_glsl as ts
    from pyevtk.hl import gridToVTK
    import time
    #from taichi_glsl import scalar

    #from taichi_glsl.scalar import isinf, isnan
    #from taichi_glsl.vector import vecFill
    #intialize taichi 
    ti.init(arch=ti.cpu)
    #ti.init(arch=ti.gpu, dynamic_index=True,offline_cache=True)

    #enable projection 
    enable_projection = True
    # 131*131*131
    nx,ny,nz = 131,131,131
    #nx,ny,nz = 131,131,131
    #external force in x,y,z direction 
    fx,fy,fz = 5.0e-5,-2e-5,0.0
    #niu = 0.1
    #liquid viscosity
    niu_l = 0.1         #psi>0
    #gas viscosity
    niu_g = 0.1         #psi<0
    #psi in color gradient calculation
    psi_solid = 0.7
    #surface tension
    CapA = 0.005 

    #Boundary condition mode: 0=periodic, 1= fix pressure, 2=fix velocity; boundary pressure value (rho); boundary velocity value for vx,vy,vz
    bc_x_left, rho_bcxl, vx_bcxl, vy_bcxl, vz_bcxl = 0, 1.0, 0.0e-5, 0.0, 0.0  #Boundary x-axis left side
    bc_x_right, rho_bcxr, vx_bcxr, vy_bcxr, vz_bcxr = 0, 0.995, 0.0, 0.0, 0.0  #Boundary x-axis right side
    bc_y_left, rho_bcyl, vx_bcyl, vy_bcyl, vz_bcyl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary y-axis left side
    bc_y_right, rho_bcyr, vx_bcyr, vy_bcyr, vz_bcyr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary y-axis right side
    bc_z_left, rho_bczl, vx_bczl, vy_bczl, vz_bczl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary z-axis left side
    bc_z_right, rho_bczr, vx_bczr, vy_bczr, vz_bczr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary z-axis right side

    bc_psi_x_left, psi_x_left = 1, -1.0          #   boundary condition for phase-field: 0 = periodic, 
    bc_psi_x_right, psi_x_right = 0, 1.0        #   1 = constant value on the boundary, value = -1.0 phase1 or 1.0 = phase 2
    bc_psi_y_left, psi_y_left = 0, 1.0
    bc_psi_y_right, psi_y_right = 0, 1.0
    bc_psi_z_left, psi_z_left = 0, 1.0
    bc_psi_z_right, psi_z_right = 0, 1.0

    # Non Sparse memory allocation
    #density distribution function nx*ny*nz*19
    f = ti.field(ti.f32,shape=(nx,ny,nz,19))
    #density distribution function nx*ny*nz*19
    F = ti.field(ti.f32,shape=(nx,ny,nz,19))
    #density nx*ny*nz
    rho = ti.field(ti.f32, shape=(nx,ny,nz))
    #velocity nx*ny*nz vector
    v = ti.Vector.field(3,ti.f32, shape=(nx,ny,nz)) 
    #psi nx*ny*nz
    psi = ti.field(ti.f32, shape=(nx,ny,nz))
    #density r nx*ny*nz
    rho_r = ti.field(ti.f32, shape=(nx,ny,nz))
    #density b nx*ny*nz
    rho_b = ti.field(ti.f32, shape=(nx,ny,nz))
    #density r nx*ny*nz
    rhor = ti.field(ti.f32, shape=(nx,ny,nz))
    #density b nx*ny*nz
    rhob = ti.field(ti.f32, shape=(nx,ny,nz))
    #lattice speed 19 dimensional vector
    e = ti.Vector.field(3,ti.i32, shape=(19))  
    #S_dig = ti.field(ti.f32,shape=(19))
    #lattice speed 19 dimensional vector
    e_f = ti.Vector.field(3,ti.f32, shape=(19))
    #weight parameter 19 dimensional vector
    w = ti.field(ti.f32, shape=(19))
    #solid flag nx*ny*nz
    solid = ti.field(ti.i32,shape=(nx,ny,nz))
    #streaming vector 19 dimensional vector
    LR = ti.field(ti.i32,shape=(19))

    #external force 3 dimensional vector
    ext_f = ti.Vector.field(3,ti.f32,shape=())
    # x-left velocity 3 dimensional vector
    bc_vel_x_left = ti.Vector.field(3,ti.f32, shape=())
    # x-right velocity 3 dimensional vector
    bc_vel_x_right = ti.Vector.field(3,ti.f32, shape=())
    # y-left velocity 3 dimensional vector
    bc_vel_y_left = ti.Vector.field(3,ti.f32, shape=())
    # y-right velocity 3 dimensional vector
    bc_vel_y_right = ti.Vector.field(3,ti.f32, shape=())
    # z-left velocity 3 dimensional vector
    bc_vel_z_left = ti.Vector.field(3,ti.f32, shape=())
    # z-right velocity 3 dimensional vector
    bc_vel_z_right = ti.Vector.field(3,ti.f32, shape=())
    #transforming matrix 19*19
    M = ti.field(ti.f32, shape=(19,19))
    #inverse transforming matrix 19*19
    inv_M = ti.field(ti.f32, shape=(19,19))
    #parameters for calculating the parameter of s diagonal 
    #=======================================#
    lg0, wl, wg = 0.0, 0.0, 0.0
    l1, l2, g1, g2 = 0.0, 0.0, 0.0, 0.0
    wl = 1.0/(niu_l/(1.0/3.0)+0.5)
    wg = 1.0/(niu_g/(1.0/3.0)+0.5)
    lg0 = 2*wl*wg/(wl+wg)
    l1=2*(wl-lg0)*10
    l2=-l1/0.2
    g1=2*(lg0-wg)*10
    g2=g1/0.2
    #=======================================#

    #transformation matrix 
    M_np = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
    [0,-2,2,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
    [0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
    [0,0,0,-2,2,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
    [0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
    [0,0,0,0,0,-2,2,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
    [0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
    [0,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
    [0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
    [0,0,0,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0],
    [0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0],
    [0,0,0,0,0,0,0,-1,1,1,-1,0,0,0,0,1,-1,1,-1],
    [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1]])
    #inverde of transforming matrix
    inv_M_np = np.linalg.inv(M_np)
    #streaming array
    LR_np = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17])
    #M matrix from the numpy 
    M.from_numpy(M_np)
    #inverse matrix from numpy
    inv_M.from_numpy(inv_M_np)

    #steaming array from numpy
    LR.from_numpy(LR_np)
    #external force with vector three dimensional
    ext_f[None] = ti.Vector([fx,fy,fz])
    #set transforming matrix, inverse matrix and streaming vector non-modified 
    ti.static(inv_M)
    ti.static(M)
    ti.static(LR)

    #set x,y,z array with nx*ny*nz
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    z = np.linspace(0, nz, nz)
    #set meshgrid and return three meshgrid matrix X,Y,Z with non-cartesian indexing 
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

``feq(k,rho_local, u)`` calculate the equilibrium denisty distribution function 

.. code-block:: python

    @ti.func
    def feq(k,rho_local, u):
        # eu=ts.vector.dot(e[k],u)
        # uv=ts.vector.dot(u,u)
        eu = e[k].dot(u)
        uv = u.dot(u)
        #same as single phase equilibrium density distribution function
        feqout = w[k]*rho_local*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
        #print(k, rho_local, w[k])
        return feqout

``init()`` intialize some variable

.. code-block:: python

    @ti.kernel
    def init():
        for i,j,k in solid:
            if (solid[i,j,k] == 0):
            #if it is fluid intialize the density and velocity be one and zero
                rho[i,j,k] = 1.0
                v[i,j,k] = ti.Vector([0,0,0])
                # set density r and density b based on psi 
                rho_r[i,j,k] = (psi[i,j,k]+1.0)/2.0
                rho_b[i,j,k] = 1.0 - rho_r[i,j,k]
                #set another density r and density b 
                rhor[i,j,k] = 0.0
                rhob[i,j,k] = 0.0
                #set density distribution equals to equilibrium density distribution function
                for s in ti.static(range(19)):
                    f[i,j,k,s] = feq(s,1.0,v[i,j,k])
                    F[i,j,k,s] = feq(s,1.0,v[i,j,k])

``init_geo(filename, filename2)`` import the geometry data

.. code-block:: python

    def init_geo(filename, filename2):
        #read the solid flag data and set it as an column major array
        in_dat = np.loadtxt(filename)
        in_dat[in_dat>0] = 1
        in_dat = np.reshape(in_dat, (nx,ny,nz),order='F')

        #read the phase data from file
        phase_in_dat = np.loadtxt(filename2)
        #set the array from the file with colum major 
        phase_in_dat = np.reshape(phase_in_dat, (nx,ny,nz), order='F')

        return in_dat, phase_in_dat

``static_init()`` initialize non-modified variable

.. code-block:: python

    @ti.kernel
    def static_init():
        if ti.static(enable_projection): # No runtime overhead
        #define lattice speed
        e[0] = ti.Vector([0,0,0])
        e[1] = ti.Vector([1,0,0]); e[2] = ti.Vector([-1,0,0]); e[3] = ti.Vector([0,1,0]); e[4] = ti.Vector([0,-1,0]);e[5] = ti.Vector([0,0,1]); e[6] = ti.Vector([0,0,-1])
        e[7] = ti.Vector([1,1,0]); e[8] = ti.Vector([-1,-1,0]); e[9] = ti.Vector([1,-1,0]); e[10] = ti.Vector([-1,1,0])
        e[11] = ti.Vector([1,0,1]); e[12] = ti.Vector([-1,0,-1]); e[13] = ti.Vector([1,0,-1]); e[14] = ti.Vector([-1,0,1])
        e[15] = ti.Vector([0,1,1]); e[16] = ti.Vector([0,-1,-1]); e[17] = ti.Vector([0,1,-1]); e[18] = ti.Vector([0,-1,1])
        #define another lattice speed 
        e_f[0] = ti.Vector([0,0,0])
        e_f[1] = ti.Vector([1,0,0]); e_f[2] = ti.Vector([-1,0,0]); e_f[3] = ti.Vector([0,1,0]); e_f[4] = ti.Vector([0,-1,0]);e_f[5] = ti.Vector([0,0,1]); e_f[6] = ti.Vector([0,0,-1])
        e_f[7] = ti.Vector([1,1,0]); e_f[8] = ti.Vector([-1,-1,0]); e_f[9] = ti.Vector([1,-1,0]); e_f[10] = ti.Vector([-1,1,0])
        e_f[11] = ti.Vector([1,0,1]); e_f[12] = ti.Vector([-1,0,-1]); e_f[13] = ti.Vector([1,0,-1]); e_f[14] = ti.Vector([-1,0,1])
        e_f[15] = ti.Vector([0,1,1]); e_f[16] = ti.Vector([0,-1,-1]); e_f[17] = ti.Vector([0,1,-1]); e_f[18] = ti.Vector([0,-1,1])
        #define a weight parameter
        w[0] = 1.0/3.0; w[1] = 1.0/18.0; w[2] = 1.0/18.0; w[3] = 1.0/18.0; w[4] = 1.0/18.0; w[5] = 1.0/18.0; w[6] = 1.0/18.0; 
        w[7] = 1.0/36.0; w[8] = 1.0/36.0; w[9] = 1.0/36.0; w[10] = 1.0/36.0; w[11] = 1.0/36.0; w[12] = 1.0/36.0; 
        w[13] = 1.0/36.0; w[14] = 1.0/36.0; w[15] = 1.0/36.0; w[16] = 1.0/36.0; w[17] = 1.0/36.0; w[18] = 1.0/36.0;
        #define the boundary velocity
        bc_vel_x_left = ti.Vector([vx_bcxl, vy_bcxl, vz_bcxl])
        bc_vel_x_right = ti.Vector([vx_bcxr, vy_bcxr, vz_bcxr])
        bc_vel_y_left = ti.Vector([vx_bcyl, vy_bcyl, vz_bcyl])
        bc_vel_y_right = ti.Vector([vx_bcyr, vy_bcyr, vz_bcyr])
        bc_vel_z_left = ti.Vector([vx_bczl, vy_bczl, vz_bczl])
        bc_vel_z_right = ti.Vector([vx_bczr, vy_bczr, vz_bczr])

``multiply_M()`` calculate the density distribution function in momentum space

.. code-block:: python

    @ti.func
    def multiply_M(i,j,k):
        out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        for index in ti.static(range(19)):
            for s in ti.static(range(19)):
                #calculate here
                out[index] += M[index,s]*F[i,j,k,s]
                #print(i,j,k, index, s, out[index], M[index,s], F[i,j,k,s])
        return out

``GuoF(i,j,k,s,u)`` calculate Guo's force term

.. code-block:: python

    @ti.func
    def GuoF(i,j,k,s,u):
        out=0.0
        for l in ti.static(range(19)):
            out += w[l]*((e_f[l]-u).dot(ext_f[None])+(e_f[l].dot(u)*(e_f[l].dot(ext_f[None]))))*M[s,l]
        
        return out

``meq_vec(rho_local,u)`` defines the equilibrium momentum 

.. code-block:: python

    @ti.func
    def meq_vec(rho_local,u):
        out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        out[0] = rho_local;             out[3] = u[0];    out[5] = u[1];    out[7] = u[2];
        out[1] = u.dot(u);    out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z;         out[11] = u.y*u.y-u.z*u.z
        out[13] = u.x*u.y;    out[14] = u.y*u.z;                            out[15] = u.x*u.z
        return out

``Compute_C()`` calculate the color gradient 

.. code-block:: python

    @ti.func
    def Compute_C(i):
        C = ti.Vector([0.0,0.0,0.0])
        ind_S = 0
        for s in ti.static(range(19)):
            ip = periodic_index_for_psi(i+e[s])
            if (solid[ip] == 0):    
                #if it's fluid calculate the color gradient based on psi
                C += 3.0*w[s]*e_f[s]*psi[ip]
            else:
                #if it is solid and abs(density r-density b) is less than 0.9 
                ind_S = 1
                #calculate the color gradient based on psi_solid and set ind_s=1
                C += 3.0*w[s]*e_f[s]*psi_solid

        if (abs(rho_r[i]-rho_b[i]) > 0.9) and (ind_S == 1):
            #if abs(density r-density b) is very large and it's solid set color gradient to be zero
            C = ti.Vector([0.0,0.0,0.0])
        
        return C

``Compute_S_local`` calculate parameter of s diagonal

.. code-block:: python

    @ti.func
    def Compute_S_local(id):
        sv=0.0; sother=0.0
        if (psi[id]>0):
            if (psi[id]>0.1): 
            #if psi>0.1   
            #sv=1.0/(niu_l/(1.0/3.0)+0.5)
                sv=wl
            else:
            #if 0<psi<0.1   calculate sv
                sv=lg0+l1*psi[id]+l2*psi[id]*psi[id]
        else:
            #if psi <-0.1
            if (psi[id]<-0.1):
            #calculate sv
                sv=wg
            else:
            #if psi >-0.1
                sv=lg0+g1*psi[id]+g2*psi[id]*psi[id]
        #calculate s other
        sother = 8.0*(2.0-sv)/(8.0-sv)

        #set s diagonal to be zero and set certain element to be relatie local parameter
        S = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        S[1]=sv;S[2]=sv;S[4]=sother;S[6]=sother;S[8]=sother;S[9]=sv;
        S[10]=sv;S[11]=sv;S[12]=sv;S[13]=sv;S[14]=sv;S[15]=sv;S[16]=sother;
        S[17]=sother;S[18]=sother;


        return S;


``collision()`` define the collision and recoloring process

.. code-block:: python

    @ti.kernel
    def colission():
        for i,j,k in rho:
            #if it is inner fluid, calculate color gradient divided by norm of color gradient
            if (i<nx and j<ny and k<nz and solid[i,j,k] == 0):
                uu = v[i,j,k].norm_sqr()
                C = Compute_C(ti.Vector([i,j,k]))
                cc = C.norm()
                normal = ti.Vector([0.0,0.0,0.0])
                if cc>0 :
                    normal = C/cc
                #calculate the M
                m_temp = multiply_M(i,j,k)
                meq = meq_vec(rho[i,j,k],v[i,j,k])
                #calculate surface tension term
                meq[1] += CapA*cc
                meq[9] += 0.5*CapA*cc*(2*normal.x*normal.x-normal.y*normal.y-normal.z*normal.z)
                meq[11] += 0.5*CapA*cc*(normal.y*normal.y-normal.z*normal.z)
                meq[13] += 0.5*CapA*cc*(normal.x*normal.y)
                meq[14] += 0.5*CapA*cc*(normal.y*normal.z)
                meq[15] += 0.5*CapA*cc*(normal.x*normal.z)
                #calculate s local
                S_local = Compute_S_local(ti.Vector([i,j,k]))
                #calculate s*(m-meq)
                for s in ti.static(range(19)):
                    m_temp[s] -= S_local[s]*(m_temp[s]-meq[s])
                    m_temp[s] += (1-0.5*S_local[s])*GuoF(i,j,k,s,v[i,j,k])
                #calculte convection of density filed 
                g_r = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                g_b = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

                for s in ti.static(range(19)):
                    f[i,j,k,s] = 0
                    for l in ti.static(range(19)):
                    # 1.single phase collision
                        f[i,j,k,s] += inv_M[s,l]*m_temp[l]

                    g_r[s] = feq(s,rho_r[i,j,k],v[i,j,k])
                    g_b[s] = feq(s,rho_b[i,j,k],v[i,j,k])

                if (cc>0):
                    for kk in ti.static([1,3,5,7,9,11,13,15,17]):
                        # ef=ts.vector.dot(e[kk],C)
                        ef=e[kk].dot(C)
                        cospsi= g_r[kk] if (g_r[kk]<g_r[kk+1]) else g_r[kk+1]
                        cospsi= cospsi if (cospsi<g_b[kk]) else g_b[kk]
                        cospsi=cospsi if (cospsi<g_b[kk+1]) else g_b[kk+1]
                        cospsi*=ef/cc
                        #2.surface tension perturbation 
                        g_r[kk]+=cospsi
                        g_r[kk+1]-=cospsi
                        g_b[kk]-=cospsi
                        g_b[kk+1]+=cospsi
                # recoloring
                for s in ti.static(range(19)):
                    ip = periodic_index(ti.Vector([i,j,k])+e[s])
                    if (solid[ip]==0):
                        rhor[ip] += g_r[s]
                        rhob[ip] += g_b[s]
                    else:
                        rhor[i,j,k] += g_r[s]
                        rhob[i,j,k] += g_b[s]

``periodic_index()`` defines the index of boundary if using periodic boundary condition

.. code-block:: python 

    @ti.func
    def periodic_index(i):
        iout = i
        if i[0]<0:     iout[0] = nx-1
        if i[0]>nx-1:  iout[0] = 0
        if i[1]<0:     iout[1] = ny-1
        if i[1]>ny-1:  iout[1] = 0
        if i[2]<0:     iout[2] = nz-1
        if i[2]>nz-1:  iout[2] = 0

        return iout

``periodic_index_for_psi(i)`` defines the index of boundary for psi if using periodic boundary condition

.. code-block:: python 

    @ti.func
    def periodic_index_for_psi(i):
        iout = i
        if i[0]<0:     
        #if periodic boundary condition set index based on periodic boundary condition
            if bc_psi_x_left == 0:  
                iout[0] = nx-1  
            else: 
        #otherwise set neighbouring index,
        #similar for other sides
                iout[0] = 0
        
        if i[0]>nx-1:
            if bc_psi_x_right==0:
                iout[0] = 0
            else:
                iout[0] = nx-1
        
        if i[1]<0:
            if bc_psi_y_left == 0:
                iout[1] = ny-1
            else:
                iout[1] = 0
        
        if i[1]>ny-1:
            if bc_psi_y_right==0:
                iout[1] = 0
            else:
                iout[1] = ny-1

        if i[2]<0:
            if bc_psi_z_left==0:
                iout[2] = nz-1
            else: 
                iout[2] = 0

        if i[2]>nz-1:
            if bc_psi_z_right==0:
                iout[2] = 0
            else:
                iout[2] = nz-1

        return iout


``streaming1()`` defines steaming process of denisty distribution function

.. code-block:: python 

    @ti.kernel
    def streaming1():
        for i,j,k in rho:
            #if (solid[i,j,k] == 0):
            if (i<nx and j<ny and k<nz and solid[i,j,k] == 0):
                ci = ti.Vector([i,j,k])
                for s in ti.static(range(19)):
                    ip = periodic_index(ci+e[s])
                    if (solid[ip]==0):
                    #if it is fluid,streaming along certain direction
                        F[ip,s] = f[ci,s]
                    else:
                    #if it is on the solid, bounce back to the opposite
                        F[ci,LR[s]] = f[ci,s]
                        #print(i, ip, "@@@")

``Boundary_condition_psi()`` defines boundary condition for psi

.. code-block:: python 

    @ti.kernel
    def Boundary_condition_psi():
        if bc_psi_x_left == 1:
            for j,k in ti.ndrange((0,ny),(0,nz)):
                if (solid[0,j,k]==0):
                #if it is fluid the value of psi equals to the psi_x_left
                    psi[0,j,k] = psi_x_left
                #calculate density according to psi 
                #similar for other sides
                    rho_r[0,j,k] = (psi_x_left + 1.0)/2.0
                    rho_b[0,j,k] = 1.0 - rho_r[0,j,k]

        if bc_psi_x_right == 1:
            for j,k in ti.ndrange((0,ny),(0,nz)):
                if (solid[nx-1,j,k]==0):
                    psi[nx-1,j,k] = psi_x_right
                    rho_r[nx-1,j,k] = (psi_x_right + 1.0)/2.0
                    rho_b[nx-1,j,k] = 1.0 - rho_r[nx-1,j,k]

        if bc_psi_y_left == 1:
            for i,k in ti.ndrange((0,nx),(0,nz)):
                if (solid[i,0,k]==0):
                    psi[i,0,k] = psi_y_left
                    rho_r[i,0,k] = (psi_y_left + 1.0)/2.0
                    rho_b[i,0,k] = 1.0 - rho_r[i,0,k]
        
        if bc_psi_y_right == 1:
            for i,k in ti.ndrange((0,nx),(0,nz)):
                if (solid[i,ny-1,k]==0):
                    psi[i,ny-1,k] = psi_y_right
                    rho_r[i,ny-1,k] = (psi_y_right + 1.0)/2.0
                    rho_b[i,ny-1,k] = 1.0 - rho_r[i,ny-1,k]

        if bc_psi_z_left == 1:
            for i,j in ti.ndrange((0,nx),(0,ny)):
                if (solid[i,j,0]==0):
                    psi[i,j,0] = psi_z_left
                    rho_r[i,j,0] = (psi_z_left + 1.0)/2.0
                    rho_b[i,j,0] = 1.0 - rho_r[i,j,0]
        
        if bc_psi_z_right == 1:
            for i,j in ti.ndrange((0,nx),(0,ny)):
                if (solid[i,j,nz-1]==0):
                    psi[i,j,nz-1] = psi_z_right
                    rho_r[i,j,nz-1] = (psi_z_right + 1.0)/2.0
                    rho_b[i,j,nz-1] = 1.0 - rho_r[i,j,nz-1]

``Boundary_condition`` defines boundary condition and the same as single_phase solver

.. code-block:: python 

    @ti.kernel
    def Boundary_condition():
        if ti.static(bc_x_left==1):
            for j,k in ti.ndrange((0,ny),(0,nz)):
                if (solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        if (solid[1,j,k]>0):
                            F[0,j,k,s]=feq(s, rho_bcxl, v[1,j,k])
                        else:
                            F[0,j,k,s]=feq(s, rho_bcxl, v[0,j,k])

        if ti.static(bc_x_left==2):
            for j,k in ti.ndrange((0,ny),(0,nz)):
                if (solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        F[0,j,k,s]=feq(LR[s], 1.0, bc_vel_x_left[None])-F[0,j,k,LR[s]]+feq(s,1.0,bc_vel_x_left[None])  

        if ti.static(bc_x_right==1):
            for j,k in ti.ndrange((0,ny),(0,nz)):
                if (solid[nx-1,j,k]==0):
                    for s in ti.static(range(19)):
                        if (solid[nx-2,j,k]>0):
                            F[nx-1,j,k,s]=feq(s, rho_bcxr, v[nx-2,j,k])
                        else:
                            F[nx-1,j,k,s]=feq(s, rho_bcxr, v[nx-1,j,k])

        if ti.static(bc_x_right==2):
            for j,k in ti.ndrange((0,ny),(0,nz)):
                if (solid[nx-1,j,k]==0):
                    for s in ti.static(range(19)):
                        F[nx-1,j,k,s]=feq(LR[s], 1.0, bc_vel_x_right[None])-F[nx-1,j,k,LR[s]]+feq(s,1.0,bc_vel_x_right[None]) 

        
        # Direction Y
        if ti.static(bc_y_left==1):
            for i,k in ti.ndrange((0,nx),(0,nz)):
                if (solid[i,0,k]==0):
                    for s in ti.static(range(19)):
                        if (solid[i,1,k]>0):
                            F[i,0,k,s]=feq(s, rho_bcyl, v[i,1,k])
                        else:
                            F[i,0,k,s]=feq(s, rho_bcyl, v[i,0,k])

        if ti.static(bc_y_left==2):
            for i,k in ti.ndrange((0,nx),(0,nz)):
                if (solid[i,0,k]==0):
                    for s in ti.static(range(19)):
                        F[i,0,k,s]=feq(LR[s], 1.0, bc_vel_y_left[None])-F[i,0,k,LR[s]]+feq(s,1.0,bc_vel_y_left[None])  

        if ti.static(bc_y_right==1):
            for i,k in ti.ndrange((0,nx),(0,nz)):
                if (solid[i,ny-1,k]==0):
                    for s in ti.static(range(19)):
                        if (solid[i,ny-2,k]>0):
                            F[i,ny-1,k,s]=feq(s, rho_bcyr, v[i,ny-2,k])
                        else:
                            F[i,ny-1,k,s]=feq(s, rho_bcyr, v[i,ny-1,k])

        if ti.static(bc_y_right==2):
            for i,k in ti.ndrange((0,nx),(0,nz)):
                if (solid[i,ny-1,k]==0):
                    for s in ti.static(range(19)):
                        F[i,ny-1,k,s]=feq(LR[s], 1.0, bc_vel_y_right[None])-F[i,ny-1,k,LR[s]]+feq(s,1.0,bc_vel_y_right[None]) 
        
        # Z direction
        if ti.static(bc_z_left==1):
            for i,j in ti.ndrange((0,nx),(0,ny)):
                if (solid[i,j,0]==0):
                    for s in ti.static(range(19)):
                        if (solid[i,j,1]>0):
                            F[i,j,0,s]=feq(s, rho_bczl, v[i,j,1])
                        else:
                            F[i,j,0,s]=feq(s, rho_bczl, v[i,j,0])

        if ti.static(bc_z_left==2):
            for i,j in ti.ndrange((0,nx),(0,ny)):
                if (solid[i,j,0]==0):
                    for s in ti.static(range(19)):
                        F[i,j,0,s]=feq(LR[s], 1.0, bc_vel_z_left[None])-F[i,j,0,LR[s]]+feq(s,1.0,bc_vel_z_left[None])  

        if ti.static(bc_z_right==1):
            for i,j in ti.ndrange((0,nx),(0,ny)):
                if (solid[i,j,nz-1]==0):
                    for s in ti.static(range(19)):
                        if (solid[i,j,nz-2]>0):
                            F[i,j,nz-1,s]=feq(s, rho_bczr, v[i,j,nz-2])
                        else:
                            F[i,j,nz-1,s]=feq(s, rho_bczr, v[i,j,nz-1])

        if ti.static(bc_z_right==2):
            for i,j in ti.ndrange((0,nx),(0,ny)):
                if (solid[i,j,nz-1]==0):
                    for s in ti.static(range(19)):
                        F[i,j,nz-1,s]=feq(LR[s], 1.0, bc_vel_z_right[None])-F[i,j,nz-1,LR[s]]+feq(s,1.0,bc_vel_z_right[None]) 

``Boundary_condition_psi()`` calculate macroscopic variable

.. code-block:: python      

    @ti.kernel
    def streaming3():
        for i,j,k, in rho:
            #if (solid[i,j,k] == 0):
            if (i<nx and j<ny and k<nz and solid[i,j,k] == 0):
                rho[i,j,k] = 0
                v[i,j,k] = ti.Vector([0,0,0])
                #define denisty r and density b
                rho_r[i,j,k] = rhor[i,j,k]
                rho_b[i,j,k] = rhob[i,j,k]
                rhor[i,j,k] = 0.0; rhob[i,j,k] = 0.0

                for s in ti.static(range(19)):
                    f[i,j,k,s] = F[i,j,k,s]
                    rho[i,j,k] += f[i,j,k,s]
                    v[i,j,k] += e_f[s]*f[i,j,k,s]
                #calculate velocity and psi
                v[i,j,k] /= rho[i,j,k]
                v[i,j,k] += (ext_f[None]/2)/rho[i,j,k]
                psi[i,j,k] = rho_r[i,j,k]-rho_b[i,j,k]/(rho_r[i,j,k] + rho_b[i,j,k])
                    
The code snippts below define time, read file do the simulation and export results
It is almost the same as the single-phase solver except two input file and export phase variable

.. code-block:: python 

    time_init = time.time()
    time_now = time.time()
    time_pre = time.time()
    dt_count = 0               


    solid_np, phase_np = init_geo('./img_ftb131.txt','./phase_ftb131.dat')

    #solid_np = init_geo('./img_ftb131.txt')
    solid.from_numpy(solid_np)
    psi.from_numpy(phase_np)

    static_init()
    init()

    #print(wl,wg, lg0, l1, l2,'~@@@@@~@~@~@~@')

    for iter in range(80000+1):
        colission()
        streaming1()
        Boundary_condition()
        #streaming2()
        streaming3()
        Boundary_condition_psi()

        
        if (iter%500==0):
            
            time_pre = time_now
            time_now = time.time()
            diff_time = int(time_now-time_pre)
            elap_time = int(time_now-time_init)
            m_diff, s_diff = divmod(diff_time, 60)
            h_diff, m_diff = divmod(m_diff, 60)
            m_elap, s_elap = divmod(elap_time, 60)
            h_elap, m_elap = divmod(m_elap, 60)
            
            print('----------Time between two outputs is %dh %dm %ds; elapsed time is %dh %dm %ds----------------------' %(h_diff, m_diff, s_diff,h_elap,m_elap,s_elap))
            print('The %dth iteration, Max Force = %f,  force_scale = %f\n\n ' %(iter, 10.0,  10.0))
            
            if (iter%10000==0):
                gridToVTK(
                    "./structured"+str(iter),
                    x,
                    y,
                    z,
                    #cellData={"pressure": pressure},
                    pointData={ "Solid": np.ascontiguousarray(solid.to_numpy()),
                                "rho": np.ascontiguousarray(rho.to_numpy()[0:nx,0:ny,0:nz]),
                                "phase": np.ascontiguousarray(psi.to_numpy()[0:nx,0:ny,0:nz]),
                                "velocity": (np.ascontiguousarray(v.to_numpy()[0:nx,0:ny,0:nz,0]), np.ascontiguousarray(v.to_numpy()[0:nx,0:ny,0:nz,1]),np.ascontiguousarray(v.to_numpy()[0:nx,0:ny,0:nz,2]))
                                }
                )   

    #ti.print_kernel_profile_info()
    #ti.print_profile_info()
