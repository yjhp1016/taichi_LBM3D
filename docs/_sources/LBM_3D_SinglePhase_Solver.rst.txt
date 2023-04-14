Single\_phase.LBM\_3D\_SinglePhase\_Solver
==============================================
This is a D3Q19 MRT(multi-relaxation-time) solver for single phase. It defines a class called ``LB3D_Solver_Single_Phase``. The Class has a default function
``__init__()`` as normal python class.

.. code-block:: python

    class LB3D_Solver_Single_Phase:
        def __init__(self, nx, ny, nz, sparse_storage = False):
        #enable projection, define a sparse_storage flag 
        self.enable_projection = True
        self.sparse_storage = sparse_storage
        #the grid of the simulation in three direction
        self.nx,self.ny,self.nz = nx,ny,nz
        #nx,ny,nz = 120,120,120
        #density distribution function in three direction
        self.fx,self.fy,self.fz = 0.0e-6,0.0,0.0
        #kinematic viscosity in lattice unit 
        self.niu = 0.16667
        #define a taichi field of float scalar which is the maximum velocity 
        self.max_v=ti.field(ti.f32,shape=())
        #Boundary condition mode: 0=periodic, 1= fix pressure, 2=fix velocity; boundary pressure value (rho); boundary velocity value for vx,vy,vz
        self.bc_x_left, self.rho_bcxl, self.vx_bcxl, self.vy_bcxl, self.vz_bcxl = 0, 1.0, 0.0e-5, 0.0, 0.0  #Boundary x-axis left side
        self.bc_x_right, self.rho_bcxr, self.vx_bcxr, self.vy_bcxr, self.vz_bcxr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis right side
        self.bc_y_left, self.rho_bcyl, self.vx_bcyl, self.vy_bcyl, self.vz_bcyl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary y-axis left side
        self.bc_y_right, self.rho_bcyr, self.vx_bcyr, self.vy_bcyr, self.vz_bcyr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary y-axis right side
        self.bc_z_left, self.rho_bczl, self.vx_bczl, self.vy_bczl, self.vz_bczl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary z-axis left side
        self.bc_z_right, self.rho_bczr, self.vx_bczr, self.vy_bczr, self.vz_bczr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary z-axis right side
        if sparse_storage == False:
            #define old density distribution function with taichi field which has nx*ny*nz element and each element is a 19 dimensional vector 
            self.f = ti.Vector.field(19,ti.f32,shape=(nx,ny,nz))
            #define new density distribution function with taichi field which has nx*ny*nz element and each element is a 19 dimensional vector 
            self.F = ti.Vector.field(19,ti.f32,shape=(nx,ny,nz))
            #define density with taichi field which has nx*ny*nz element and each element is a scalar 
            self.rho = ti.field(ti.f32, shape=(nx,ny,nz))
            #define velocity with taichi field which has nx*ny*nz element and each element is a three dimensional vector 
            self.v = ti.Vector.field(3,ti.f32, shape=(nx,ny,nz))
        else:
            #sparse storage the variable
            #define old density distribution function by taichi field with one element and which is a 19 dimensional vector 
            self.f = ti.Vector.field(19, ti.f32)
            #define new density distribution function by taichi field with one element and which is a 19 dimensional vector 
            self.F = ti.Vector.field(19,ti.f32)
            #define density by taichi field with one element which is a scalar 
            self.rho = ti.field(ti.f32)
            #define velocity by taichi field with one element which is a scalar
            self.v = ti.Vector.field(3, ti.f32)
            #define partition equals 3
            n_mem_partition = 3
            #every index has four variable rho, v, f, F
            cell1 = ti.root.pointer(ti.ijk, (nx//n_mem_partition+1,ny//n_mem_partition+1,nz//n_mem_partition+1))
            cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.rho, self.v, self.f, self.F)
        #define lattice speed 3x19
        self.e = ti.Vector.field(3,ti.i32, shape=(19))
        #define s diagnol vector 
        self.S_dig = ti.Vector.field(19,ti.f32,shape=())
        #define another lattice speed 3x19 
        self.e_f = ti.Vector.field(3,ti.f32, shape=(19))
        #define weight parameter
        self.w = ti.field(ti.f32, shape=(19))
        #define solid which is a flag when equals 0 it is fluid, when it is 1 it is solid
        self.solid = ti.field(ti.i8,shape=(nx,ny,nz))
        #define external force which is a three dimensional vector
        self.ext_f = ti.Vector.field(3,ti.f32,shape=())
        #define transforming matrix M which is a 19x19 dimension matrix
        self.M = ti.Matrix.field(19, 19, ti.f32, shape=())
        #define the inverse transforming matrix M^-1
        self.inv_M = ti.Matrix.field(19,19,ti.f32, shape=())
        #define the numpy version of M.
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
        #define the numpy version of M^-1
        inv_M_np = np.linalg.inv(M_np)
        #define the index of 19 lattice node for bounce back
        self.LR = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17]
        #define taichi field version of M
        self.M[None] = ti.Matrix([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
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
        #define taichi field version of M^-1
        self.inv_M[None] = ti.Matrix(inv_M_np)
        #define coordinate nx*ny*nz
        self.x = np.linspace(0, nx, nx)
        self.y = np.linspace(0, ny, ny)
        self.z = np.linspace(0, nz, nz)
        #X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

Following is the ``init_simulation()`` function which initialize some simulation parameter

.. code-block:: python

    def init_simulation(self):
    #x,y,z velocity vector from vx_bcxl,vy_bcxl and vz_bcxl
    self.bc_vel_x_left = [self.vx_bcxl, self.vy_bcxl, self.vz_bcxl]
    self.bc_vel_x_right = [self.vx_bcxr, self.vy_bcxr, self.vz_bcxr]
    self.bc_vel_y_left = [self.vx_bcyl, self.vy_bcyl, self.vz_bcyl]
    self.bc_vel_y_right = [self.vx_bcyr, self.vy_bcyr, self.vz_bcyr]
    self.bc_vel_z_left = [self.vx_bczl, self.vy_bczl, self.vz_bczl]
    self.bc_vel_z_right = [self.vx_bczr, self.vy_bczr, self.vz_bczr]
    #define single relaxation time tau
    self.tau_f=3.0*self.niu+0.5
    #define single relaxation frequency
    self.s_v=1.0/self.tau_f
    #define other parameter in the s diagonal 
    self.s_other=8.0*(2.0-self.s_v)/(8.0-self.s_v)
    #define the s diagonal 
    self.S_dig[None] = ti.Vector([0,self.s_v,self.s_v,0,self.s_other,0,self.s_other,0,self.s_other, self.s_v, self.s_v,self.s_v,self.s_v,self.s_v,self.s_v,self.s_v,self.s_other,self.s_other,self.s_other])
    #define external force
    #self.ext_f[None] = ti.Vector([self.fx,self.fy,self.fz])
    self.ext_f[None][0] = self.fx
    self.ext_f[None][1] = self.fy
    self.ext_f[None][2] = self.fz 
    #if external force greater than zero define force_flag equals 1
    #other wise force_flag equals 0
    if ((abs(self.fx)>0) or (abs(self.fy)>0) or (abs(self.fz)>0)):
        self.force_flag = 1
    else:
        self.force_flag = 0

    #define M M^-1 S diagonal not been modified.
    ti.static(self.inv_M)
    ti.static(self.M)
    #ti.static(LR)
    ti.static(self.S_dig)
    #statically initialize 
    self.static_init()
    self.init()

``feq()`` calculate the equilibrium density distribution function in velocity space 

.. code-block:: python

    #taichi function
    @ti.func
        def feq(self, k,rho_local, u):
            eu = self.e[k].dot(u)
            uv = u.dot(u)
            #calculate the equilibrium density distribution function 
            feqout = self.w[k]*rho_local*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
            #print(k, rho_local, self.w[k])
            return feqout

``init()`` initialize density velocity and density distribution function 

.. code-block:: python
    
    @ti.kernel
    def init(self):
        for i,j,k in self.solid:
            #print(i,j,k)
            if (self.sparse_storage==False or self.solid[i,j,k]==0):
                #if it is fluid then initialize density equals one
                self.rho[i,j,k] = 1.0
                #initialize the velocity to be zero in all the direction
                self.v[i,j,k] = ti.Vector([0,0,0])
                for s in ti.static(range(19)):
                    #initialize 19 denisty distribution function equals the equilibrium density distribution function
                    self.f[i,j,k][s] = self.feq(s,1.0,self.v[i,j,k])
                    self.F[i,j,k][s] = self.feq(s,1.0,self.v[i,j,k])
                    #print(F[i,j,k,s], feq(s,1.0,v[i,j,k]))

``init_geo()`` import data from a file 

.. code-block:: python

    def init_geo(self,filename):
        #load data from a file
        in_dat = np.loadtxt(filename)
        #set any positive value to be one 
        in_dat[in_dat>0] = 1
        #reshape it as a nx*ny*nz vector with column major
        in_dat = np.reshape(in_dat, (self.nx,self.ny,self.nz),order='F')
        #assign it to solid varible
        self.solid.from_numpy(in_dat)

``static_init()`` initialize lattice speeed and weight parameter. These parameter is not modified during the simulation

.. code-block:: python

    #taichi kernel for parallization
    @ti.kernel
    def static_init(self):
        if ti.static(self.enable_projection): # No runtime overhead
            #initialize the lattice speed 
            self.e[0] = ti.Vector([0,0,0])
            self.e[1] = ti.Vector([1,0,0]); self.e[2] = ti.Vector([-1,0,0]); self.e[3] = ti.Vector([0,1,0]); self.e[4] = ti.Vector([0,-1,0]);self.e[5] = ti.Vector([0,0,1]); self.e[6] = ti.Vector([0,0,-1])
            self.e[7] = ti.Vector([1,1,0]); self.e[8] = ti.Vector([-1,-1,0]); self.e[9] = ti.Vector([1,-1,0]); self.e[10] = ti.Vector([-1,1,0])
            self.e[11] = ti.Vector([1,0,1]); self.e[12] = ti.Vector([-1,0,-1]); self.e[13] = ti.Vector([1,0,-1]); self.e[14] = ti.Vector([-1,0,1])
            self.e[15] = ti.Vector([0,1,1]); self.e[16] = ti.Vector([0,-1,-1]); self.e[17] = ti.Vector([0,1,-1]); self.e[18] = ti.Vector([0,-1,1])

            self.e_f[0] = ti.Vector([0,0,0])
            self.e_f[1] = ti.Vector([1,0,0]); self.e_f[2] = ti.Vector([-1,0,0]); self.e_f[3] = ti.Vector([0,1,0]); self.e_f[4] = ti.Vector([0,-1,0]);self.e_f[5] = ti.Vector([0,0,1]); self.e_f[6] = ti.Vector([0,0,-1])
            self.e_f[7] = ti.Vector([1,1,0]); self.e_f[8] = ti.Vector([-1,-1,0]); self.e_f[9] = ti.Vector([1,-1,0]); self.e_f[10] = ti.Vector([-1,1,0])
            self.e_f[11] = ti.Vector([1,0,1]); self.e_f[12] = ti.Vector([-1,0,-1]); self.e_f[13] = ti.Vector([1,0,-1]); self.e_f[14] = ti.Vector([-1,0,1])
            self.e_f[15] = ti.Vector([0,1,1]); self.e_f[16] = ti.Vector([0,-1,-1]); self.e_f[17] = ti.Vector([0,1,-1]); self.e_f[18] = ti.Vector([0,-1,1])
            #initialize the weight parameter
            self.w[0] = 1.0/3.0; self.w[1] = 1.0/18.0; self.w[2] = 1.0/18.0; self.w[3] = 1.0/18.0; self.w[4] = 1.0/18.0; self.w[5] = 1.0/18.0; self.w[6] = 1.0/18.0; 
            self.w[7] = 1.0/36.0; self.w[8] = 1.0/36.0; self.w[9] = 1.0/36.0; self.w[10] = 1.0/36.0; self.w[11] = 1.0/36.0; self.w[12] = 1.0/36.0; 
            self.w[13] = 1.0/36.0; self.w[14] = 1.0/36.0; self.w[15] = 1.0/36.0; self.w[16] = 1.0/36.0; self.w[17] = 1.0/36.0; self.w[18] = 1.0/36.0;

``meq_vec(self, rho_local,u)`` defines the equilibrium momentum 

.. code-block:: python

    @ti.func
    def meq_vec(self, rho_local,u):
        out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        out[0] = rho_local;             out[3] = u[0];    out[5] = u[1];    out[7] = u[2];
        out[1] = u.dot(u);    out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z;         out[11] = u.y*u.y-u.z*u.z
        out[13] = u.x*u.y;    out[14] = u.y*u.z;                            out[15] = u.x*u.z
        return out

``cal_local_force(self,i,j,k)`` transfer the external force to a vector 

.. code-block:: python

    @ti.func
    def cal_local_force(self,i,j,k):
        f = ti.Vector([self.fx, self.fy, self.fz])
        return f

``collision()`` defines the collision of LBM process

.. code-block:: python

    #taichi kernel for parallization
    @ti.kernel
    def colission(self):
        #outer loop for every index in rho field
        for i,j,k in self.rho:
            #if is not solid and it is not on the boundary
            if (self.solid[i,j,k] == 0 and i<self.nx and j<self.ny and k<self.nz):
                #calculate S*(m-meq)
                m_temp = self.M[None]@self.F[i,j,k]
                meq = self.meq_vec(self.rho[i,j,k],self.v[i,j,k])
                m_temp -= self.S_dig[None]*(m_temp-meq)
                #add force if there is force, here use Guo's force scheme
                f = self.cal_local_force(i,j,k)
                if (ti.static(self.force_flag==1)):
                    for s in ti.static(range(19)):
                    #    m_temp[s] -= S_dig[s]*(m_temp[s]-meq[s])
                        #f = self.cal_local_force()
                        f_guo=0.0
                        for l in ti.static(range(19)):
                            f_guo += self.w[l]*((self.e_f[l]-self.v[i,j,k]).dot(f)+(self.e_f[l].dot(self.v[i,j,k])*(self.e_f[l].dot(f))))*self.M[None][s,l]
                        #m_temp[s] += (1-0.5*self.S_dig[None][s])*self.GuoF(i,j,k,s,self.v[i,j,k],force)
                        m_temp[s] += (1-0.5*self.S_dig[None][s])*f_guo
                #calculate density distribution function after collision f=M^-1*S*(m-meq)
                self.f[i,j,k] = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                self.f[i,j,k] += self.inv_M[None]@m_temp

``periodic_index(self,i)`` defines the index of boundary if using periodic boundary condition

.. code-block:: python

    @ti.func
    def periodic_index(self,i):
        iout = i
        #x-left
        if i[0]<0:     iout[0] = self.nx-1
        #x-right
        if i[0]>self.nx-1:  iout[0] = 0
        #y-left
        if i[1]<0:     iout[1] = self.ny-1
        #y-right
        if i[1]>self.ny-1:  iout[1] = 0
        #z-left
        if i[2]<0:     iout[2] = self.nz-1
        #z-right
        if i[2]>self.nz-1:  iout[2] = 0

        return iout

``streaming1()`` defines the streaming prcoess of denisty distribution function

.. code-block:: python

    #taichi kernel for parallization
    @ti.kernel
    def streaming1(self):
        #grouped index which loop the index of rho
        for i in ti.grouped(self.rho):
        # streaming for fluid and non-boundary 
            if (self.solid[i] == 0 and i.x<self.nx and i.y<self.ny and i.z<self.nz):
                for s in ti.static(range(19)):
                # streaming according to the lattice speed and on boundary with periodic index
                    ip = self.periodic_index(i+self.e[s])
                    if (self.solid[ip]==0):
                    # fluid new density distribution function equals the streaming of old density distribution fuction
                        self.F[ip][s] = self.f[i][s]
                    else:
                    #solid bounce back scheme
                        self.F[i][self.LR[s]] = self.f[i][s]
                        #print(i, ip, "@@@")

``Boundary_condition()`` define three direction fixed pressure or fixed velocity bounary condition

.. code-block:: python

    @ti.kernel
    def Boundary_condition(self):
    #fixed pressure boundary condition 
        if ti.static(self.bc_x_left==1):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[1,j,k]>0):
                        # if the boundary is fluid but the neighbour is solid then the density distribution 
                        #function equals to the solid velcity equilibrium density distribution fucntion 
                            self.F[0,j,k][s]=self.feq(s, self.rho_bcxl, self.v[1,j,k])
                        else:
                        # if the boundary is fluid and the neighbour is fluid then the density distribution 
                        #function equals to equilibrium density distribution fucntion on the boundary 
                            self.F[0,j,k][s]=self.feq(s, self.rho_bcxl, self.v[0,j,k])
        #fixed velocity boundary condition
        if ti.static(self.bc_x_left==2):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
            # if the boundary is fluid new density distribution fucntion equals to equilibrium density
            #distibution function with fixed velocity 
                if (self.solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        #F[0,j,k][s]=feq(LR[s], 1.0, bc_vel_x_left[None])-F[0,j,k,LR[s]]+feq(s,1.0,bc_vel_x_left[None])  #!!!!!!change velocity in feq into vector
                        self.F[0,j,k][s]=self.feq(s,1.0,ti.Vector(self.bc_vel_x_left))
        # fixed pressure boundary condition on x-right similar for x-left
        if ti.static(self.bc_x_right==1):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[self.nx-1,j,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[self.nx-2,j,k]>0):
                            self.F[self.nx-1,j,k][s]=self.feq(s, self.rho_bcxr, self.v[self.nx-2,j,k])
                        else:
                            self.F[self.nx-1,j,k][s]=self.feq(s, self.rho_bcxr, self.v[self.nx-1,j,k])
        # fixed velocity boubndary condition on x-right similar for x-left
        if ti.static(self.bc_x_right==2):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[self.nx-1,j,k]==0):
                    for s in ti.static(range(19)):
                        #F[nx-1,j,k][s]=feq(LR[s], 1.0, bc_vel_x_right[None])-F[nx-1,j,k,LR[s]]+feq(s,1.0,bc_vel_x_right[None])  #!!!!!!change velocity in feq into vector
                        self.F[self.nx-1,j,k][s]=self.feq(s,1.0,ti.Vector(self.bc_vel_x_right))

         # Direction Y
         #fixed pressure boundary condition on y-left similar for x direction 
        if ti.static(self.bc_y_left==1):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,0,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[i,1,k]>0):
                            self.F[i,0,k][s]=self.feq(s, self.rho_bcyl, self.v[i,1,k])
                        else:
                            self.F[i,0,k][s]=self.feq(s, self.rho_bcyl, self.v[i,0,k])
        #fixed velocity boundary condition on y-left similar for x direction 
        if ti.static(self.bc_y_left==2):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,0,k]==0):
                    for s in ti.static(range(19)):
                        #self.F[i,0,k][s]=self.feq(self.LR[s], 1.0, self.bc_vel_y_left[None])-self.F[i,0,k][LR[s]]+self.feq(s,1.0,self.bc_vel_y_left[None])
                        self.F[i,0,k][s]=self.feq(s,1.0,ti.Vector(self.bc_vel_y_left))  
        #fixed pressure boundary condition on y-right similar for x direction
        if ti.static(self.bc_y_right==1):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,self.ny-1,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[i,self.ny-2,k]>0):
                            self.F[i,self.ny-1,k][s]=self.feq(s, self.rho_bcyr, self.v[i,self.ny-2,k])
                        else:
                            self.F[i,self.ny-1,k][s]=self.feq(s, self.rho_bcyr, self.v[i,self.ny-1,k])
        #fixed velocity boundary condition on y-right similar for x direction 
        if ti.static(self.bc_y_right==2):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,self.ny-1,k]==0):
                    for s in ti.static(range(19)):
                        #self.F[i,self.ny-1,k][s]=self.feq(self.LR[s], 1.0, self.bc_vel_y_right[None])-self.F[i,self.ny-1,k][self.LR[s]]+self.feq(s,1.0,self.bc_vel_y_right[None]) 
                        self.F[i,self.ny-1,k][s]=self.feq(s,1.0,ti.Vector(self.bc_vel_y_right))

        # Z direction
        #fixed pressure boundary condition on z-left similar for x direction 
        if ti.static(self.bc_z_left==1):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,0]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[i,j,1]>0):
                            self.F[i,j,0][s]=self.feq(s, self.rho_bczl, self.v[i,j,1])
                        else:
                            self.F[i,j,0][s]=self.feq(s, self.rho_bczl, self.v[i,j,0])
        #fixed velocity boundary condition on z-left similar for x direction 
        if ti.static(self.bc_z_left==2):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,0]==0):
                    for s in ti.static(range(19)):
                        #self.F[i,j,0][s]=self.feq(self.LR[s], 1.0, self.bc_vel_z_left[None])-self.F[i,j,0][self.LR[s]]+self.feq(s,1.0,self.bc_vel_z_left[None])  
                        self.F[i,j,0][s]=self.feq(s,1.0,ti.Vector(self.bc_vel_z_left))
        #fixed pressure boundary condition on z-right similar for x direction 
        if ti.static(self.bc_z_right==1):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,self.nz-1]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[i,j,self.nz-2]>0):
                            self.F[i,j,self.nz-1,s]=self.feq(s, self.rho_bczr, self.v[i,j,self.nz-2])
                        else:
                            self.F[i,j,self.nz-1][s]=self.feq(s, self.rho_bczr, self.v[i,j,self.nz-1])
        #fixed velocity boundary condition on z-right similar for x direction 
        if ti.static(self.bc_z_right==2):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,self.nz-1]==0):
                    for s in ti.static(range(19)):
                        #self.F[i,j,self.nz-1][s]=self.feq(self.LR[s], 1.0, self.bc_vel_z_right[None])-self.F[i,j,self.nz-1][self.LR[s]]+self.feq(s,1.0,self.bc_vel_z_right[None]) 
                        self.F[i,j,self.nz-1][s]=self.feq(s,1.0,ti.Vector(self.bc_vel_z_right))

``streaming3()`` calculatet the macroscopic variable 

.. code-block:: python

    @ti.kernel
    def streaming3(self):
        for i in ti.grouped(self.rho):
            #print(i.x, i.y, i.z)
            #if it is fluid and not on the boundary 
            if (self.solid[i]==0 and i.x<self.nx and i.y<self.ny and i.z<self.nz):
                self.rho[i] = 0
                self.v[i] = ti.Vector([0,0,0])
                self.f[i] = self.F[i]
                #calculate density 
                self.rho[i] += self.f[i].sum()

                for s in ti.static(range(19)):
                    self.v[i] += self.e_f[s]*self.f[i][s]
                
                f = self.cal_local_force(i.x, i.y, i.z)

                self.v[i] /= self.rho[i]
                #calculate velocity 
                self.v[i] += (f/2)/self.rho[i]
                
            else:
            # if it is solid the velocity is zero and the density equals one
                self.rho[i] = 1.0
                self.v[i] = ti.Vector([0,0,0])

these function set bnoundary velocity, set viscosity,force and get and calculate maximum velocity

.. code-block:: python

    #get maxium velocity
    def get_max_v(self):
        self.max_v[None] = -1e10
        self.cal_max_v()
        return self.max_v[None]

    #calculate maximum velocity with taichi kernel
    @ti.kernel
    def cal_max_v(self):
        for I in ti.grouped(self.rho):
            ti.atomic_max(self.max_v[None], self.v[I].norm())

    #set x-right velocity 
    def set_bc_vel_x1(self, vel):
        self.bc_x_right = 2
        self.vx_bcxr = vel[0]; self.vy_bcxr = vel[1]; self.vz_bcxr = vel[2];
    #set x-left velocity 
    def set_bc_vel_x0(self, vel):
        self.bc_x_left = 2
        self.vx_bcxl = vel[0]; self.vy_bcxl = vel[1]; self.vz_bcxl = vel[2];
    #set y-right velocity
    def set_bc_vel_y1(self, vel):
        self.bc_y_right = 2
        self.vx_bcyr = vel[0]; self.vy_bcyr = vel[1]; self.vz_bcyr = vel[2];
    #set y-left velocity 
    def set_bc_vel_y0(self, vel):
        self.bc_y_left = 2
        self.vx_bcyl = vel[0]; self.vy_bcyl = vel[1]; self.vz_bcyl = vel[2];
    #set z-right velocity
    def set_bc_vel_z1(self, vel):
        self.bc_z_right = 2
        self.vx_bczr = vel[0]; self.vy_bczr = vel[1]; self.vz_bczr = vel[2];
    #set z-left velocity 
    def set_bc_vel_z0(self, vel):
        self.bc_z_left = 2
        self.vx_bczl = vel[0]; self.vy_bczl = vel[1]; self.vz_bczl = vel[2];  
    #set x-left density                 
    def set_bc_rho_x0(self, rho):
        self.bc_x_left = 1
        self.rho_bcxl = rho
    #set x-right density
    def set_bc_rho_x1(self, rho):
        self.bc_x_right = 1
        self.rho_bcxr = rho
    #set y-left density 
    def set_bc_rho_y0(self, rho):
        self.bc_y_left = 1
        self.rho_bcyl = rho
    #set y-right density
    def set_bc_rho_y1(self, rho):
        self.bc_y_right = 1
        self.rho_bcyr = rho
    #set z-left density 
    def set_bc_rho_z0(self, rho):
        self.bc_z_left = 1
        self.rho_bczl = rho
    #set z-right density 
    def set_bc_rho_z1(self, rho):
        self.bc_z_right = 1
        self.rho_bczr = rho

    #set viscosity 
    def set_viscosity(self,niu):
        self.niu = niu
    #set external force
    def set_force(self,force):
        self.fx = force[0]; self.fy = force[1]; self.fz = force[2];

``export_VTK(self, n)`` function export results to vtk file use the package pyevtk

.. code-block:: python

    def export_VTK(self, n):
    #the function takes three arguments: the filename,coordinate system and the dictionary for reuslts
        gridToVTK(
            #file name
                "./LB_SingelPhase_"+str(n),
            #coordinate
                self.x,
                self.y,
                self.z,
                #cellData={"pressure": pressure},
            #the three dictionary which the key is solid,rho,velocity and it will be output to the vtk file
                pointData={ "Solid": np.ascontiguousarray(self.solid.to_numpy()),
                            "rho": np.ascontiguousarray(self.rho.to_numpy()),
                            "velocity": (   np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,0]), 
                                            np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,1]),
                                            np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,2]))
                            }
            )   

``step()`` function define the simulation process of this solver

.. code-block:: python

    def step(self):
        self.colission()
        self.streaming1()
        self.Boundary_condition()
        self.streaming3()
