# NOTE: THIS CODE NEED taichi_glsl, so please use taichi version <=0.8.5
import taichi as ti
import numpy as np
#import taichi_glsl as ts
from pyevtk.hl import gridToVTK
import time
#from taichi_glsl import scalar

#from taichi_glsl.scalar import isinf, isnan
#from taichi_glsl.vector import vecFill

ti.init(arch=ti.cpu)
#ti.init(arch=ti.gpu, dynamic_index=True,offline_cache=True)
@ti.data_oriented
class LB3D_Solver_two_Phase:
    def __init__(self, nx, ny, nz, sparse_storage = False):

        self.enable_projection = True
        self.sparse_storage = sparse_storage
        self.max_v= ti.field(ti.f32,shape=())
        self.nx,self.ny,self.nz = nx,ny,nz
        #nx,ny,nz = 131,131,131
        self.fx,self.fy,self.fz = 5.0e-5,-2e-5,0.0
        #niu = 0.1
        self.niu_l = 0.1         #psi>0
        self.niu_g = 0.1         #psi<0
        self.psi_solid = 0.7
        self.CapA = 0.005

    #Boundary condition mode: 0=periodic, 1= fix pressure, 2=fix velocity; boundary pressure value (rho); boundary velocity value for vx,vy,vz
        self.bc_x_left, self.rho_bcxl, self.vx_bcxl, self.vy_bcxl, self.vz_bcxl = 0, 1.0, 0.0e-5, 0.0, 0.0  #Boundary x-axis left side
        self.bc_x_right, self.rho_bcxr, self.vx_bcxr, self.vy_bcxr, self.vz_bcxr = 0, 0.995, 0.0, 0.0, 0.0  #Boundary x-axis left side
        self.bc_y_left, self.rho_bcyl, self.vx_bcyl, self.vy_bcyl, self.vz_bcyl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
        self.bc_y_right, self.rho_bcyr, self.vx_bcyr, self.vy_bcyr, self.vz_bcyr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
        self.bc_z_left, self.rho_bczl, self.vx_bczl, self.vy_bczl, self.vz_bczl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
        self.bc_z_right, self.rho_bczr, self.vx_bczr, self.vy_bczr, self.vz_bczr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side

        self.bc_psi_x_left, self.psi_x_left = 1, -1.0          #   boundary condition for phase-field: 0 = periodic, 
        self.bc_psi_x_right, self.psi_x_right = 0, 1.0        #   1 = constant value on the boundary, value = -1.0 phase1 or 1.0 = phase 2
        self.bc_psi_y_left, self.psi_y_left = 0, 1.0
        self.bc_psi_y_right, self.psi_y_right = 0, 1.0
        self.bc_psi_z_left, self.psi_z_left = 0, 1.0
        self.bc_psi_z_right, self.psi_z_right = 0, 1.0

    # Non Sparse memory allocation
        if sparse_storage == False:
            self.f = ti.field(ti.f32,shape=(nx,ny,nz,19))
            self.F = ti.field(ti.f32,shape=(nx,ny,nz,19))
            self.rho = ti.field(ti.f32, shape=(nx,ny,nz))
            self.v = ti.Vector.field(3,ti.f32, shape=(nx,ny,nz)) 

            self.psi = ti.field(ti.f32, shape=(nx,ny,nz))
            self.rho_r = ti.field(ti.f32, shape=(nx,ny,nz))
            self.rho_b = ti.field(ti.f32, shape=(nx,ny,nz))
            self.rhor = ti.field(ti.f32, shape=(nx,ny,nz))
            self.rhob = ti.field(ti.f32, shape=(nx,ny,nz))
        else:
            # Sparse Storage memory allocation
            self.f = ti.field(ti.f32)
            self.F = ti.field(ti.f32)
            self.rho = ti.field(ti.f32)
            self.v = ti.Vector.field(3, ti.f32)
            n_mem_partition = 3

    #-------------------------------------------------------
            cell1 = ti.root.pointer(ti.ijk, (nx//n_mem_partition+1,ny//n_mem_partition+1,nz//n_mem_partition+1))
            cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.rho)
            cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.psi)
            cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.rhor)
            cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.rhob)
            cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.rho_r)
            cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.rho_b)
            cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.v)

            cell2 = ti.root.pointer(ti.ijkl,(nx//3+1,ny//3+1,nz//3+1,1))
            cell2.dense(ti.ijkl,(n_mem_partition,n_mem_partition,n_mem_partition,19)).place(self.f)
            cell2.dense(ti.ijkl,(n_mem_partition,n_mem_partition,n_mem_partition,19)).place(self.F)


        self.e = ti.Vector.field(3,ti.i32, shape=(19))  
        #S_dig = ti.field(ti.f32,shape=(19))
        self.e_f = ti.Vector.field(3,ti.f32, shape=(19))
        self.w = ti.field(ti.f32, shape=(19))
        self.solid = ti.field(ti.i32,shape=(nx,ny,nz))

        self.ext_f = ti.Vector.field(3,ti.f32,shape=())
        # self.bc_vel_x_left = ti.Vector.field(3,ti.f32, shape=())
        # self.bc_vel_x_right = ti.Vector.field(3,ti.f32, shape=())
        # self.bc_vel_y_left = ti.Vector.field(3,ti.f32, shape=())
        # self.bc_vel_y_right = ti.Vector.field(3,ti.f32, shape=())
        # self.bc_vel_z_left = ti.Vector.field(3,ti.f32, shape=())
        # self.bc_vel_z_right = ti.Vector.field(3,ti.f32, shape=())

        self.M = ti.field(ti.f32, shape=(19,19))
        self.inv_M = ti.field(ti.f32, shape=(19,19))

    #tau_f=3.0*niu+0.5
    #s_v=1.0/tau_f
    #s_other=8.0*(2.0-s_v)/(8.0-s_v)
        self.lg0, self.wl, self.wg = 0.0, 0.0, 0.0
        self.l1, self.l2, self.g1, self.g2 =0.0, 0.0, 0.0, 0.0
        self.w1 = 1.0/(self.niu_l/(1.0/3.0)+0.5)
        self.wg = 1.0/(self.niu_g/(1.0/3.0)+0.5)
        self.lg0 = 2*self.wl*self.wg/(self.wl+self.wg)
        self.l1=2*(self.wl-self.lg0)*10
        self.l2=-self.l1/0.2
        self.g1=2*(self.lg0-self.wg)*10
        self.g2=self.g1/0.2
        # self.M = ti.Matrix.field(19, 19, ti.f32)
        # self.inv_M = ti.Matrix.field(19,19,ti.f32)

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
        inv_M_np = np.linalg.inv(M_np)

        self.LR = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17]


        # self.M = ti.Matrix([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        # [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
        # [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1],
        # [0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
        # [0,-2,2,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
        # [0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
        # [0,0,0,-2,2,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
        # [0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
        # [0,0,0,0,0,-2,2,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
        # [0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
        # [0,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
        # [0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
        # [0,0,0,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
        # [0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0,0,0,0,0],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1],
        # [0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0],
        # [0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0],
        # [0,0,0,0,0,0,0,-1,1,1,-1,0,0,0,0,1,-1,1,-1],
        # [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1]])

        self.LR = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17]
        # self.inv_M[None] = ti.Matrix(inv_M_np)
        self.M.from_numpy(M_np)
        self.inv_M.from_numpy(inv_M_np)

        # self.LR.from_numpy(LR_np)
        self.ext_f[None] = ti.Vector([self.fx,self.fy,self.fz])
        #S_dig.from_numpy(S_dig_np)

        #print(S_dig_np)
        # ti.static(inv_M)
        # ti.static(M)
        # ti.static(LR)
        #ti.static(S_dig)


        self.x = np.linspace(0, nx, nx)
        self.y = np.linspace(0, ny, ny)
        self.z = np.linspace(0, nz, nz)
        # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    def init_simulation(self):
        self.bc_vel_x_left = [self.vx_bcxl, self.vy_bcxl, self.vz_bcxl]
        self.bc_vel_x_right = [self.vx_bcxr, self.vy_bcxr, self.vz_bcxr]
        self.bc_vel_y_left = [self.vx_bcyl, self.vy_bcyl, self.vz_bcyl]
        self.bc_vel_y_right = [self.vx_bcyr, self.vy_bcyr, self.vz_bcyr]
        self.bc_vel_z_left = [self.vx_bczl, self.vy_bczl, self.vz_bczl]
        self.bc_vel_z_right = [self.vx_bczr, self.vy_bczr, self.vz_bczr]

        # self.s_v=1.0/self.tau_f
        # self.s_other=8.0*(2.0-self.s_v)/(8.0-self.s_v)

        # self.S_dig[None] = ti.Vector([0,self.s_v,self.s_v,0,self.s_other,0,self.s_other,0,self.s_other, self.s_v, self.s_v,self.s_v,self.s_v,self.s_v,self.s_v,self.s_v,self.s_other,self.s_other,self.s_other])

        self.ext_f[None][0] = self.fx
        self.ext_f[None][1] = self.fy
        self.ext_f[None][2] = self.fz 

        ti.static(self.inv_M)
        ti.static(self.M)
        ti.static(self.LR)
        self.static_init()
        self.init()

    @ti.func
    def feq(self,k,rho_local, u):
        # eu=ts.vector.dot(e[k],u)
        # uv=ts.vector.dot(u,u)
        eu = self.e[k].dot(u)
        uv = u.dot(u)
        feqout = self.w[k]*rho_local*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
        #print(k, rho_local, w[k])
        return feqout


    @ti.kernel
    def init(self):
        for i,j,k in self.solid:
            if (self.sparse_storage==False or self.solid[i,j,k]==0):
                self.rho[i,j,k] = 1.0
                self.v[i,j,k] = ti.Vector([0,0,0])

                self.rho_r[i,j,k] = (self.psi[i,j,k]+1.0)/2.0
                self.rho_b[i,j,k] = 1.0 - self.rho_r[i,j,k]
                self.rhor[i,j,k] = 0.0
                self.rhob[i,j,k] = 0.0

                for s in ti.static(range(19)):
                    self.f[i,j,k,s] = self.feq(s,1.0,self.v[i,j,k])
                    self.F[i,j,k,s] = self.feq(s,1.0,self.v[i,j,k])

                    
                    #print(F[i,j,k,s], feq(s,1.0,v[i,j,k]))

    def init_geo(self,filename, filename2):
        in_dat = np.loadtxt(filename)
        in_dat[in_dat>0] = 1
        in_dat = np.reshape(in_dat, (self.nx,self.ny,self.nz),order='F')

        phase_in_dat = np.loadtxt(filename2)
        phase_in_dat = np.reshape(phase_in_dat, (self.nx,self.ny,self.nz), order='F')

        return in_dat, phase_in_dat

    @ti.kernel
    def static_init(self):
        if ti.static(self.enable_projection): # No runtime overhead
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

            self.w[0] = 1.0/3.0; self.w[1] = 1.0/18.0; self.w[2] = 1.0/18.0; self.w[3] = 1.0/18.0; self.w[4] = 1.0/18.0; self.w[5] = 1.0/18.0; self.w[6] = 1.0/18.0; 
            self.w[7] = 1.0/36.0; self.w[8] = 1.0/36.0; self.w[9] = 1.0/36.0; self.w[10] = 1.0/36.0; self.w[11] = 1.0/36.0; self.w[12] = 1.0/36.0; 
            self.w[13] = 1.0/36.0; self.w[14] = 1.0/36.0; self.w[15] = 1.0/36.0; self.w[16] = 1.0/36.0; self.w[17] = 1.0/36.0; self.w[18] = 1.0/36.0;


    @ti.func
    def multiply_M(self, i, j, k):
        out = ti.Vector([0.0] * 19)
        for index in ti.static(range(19)):
            for s in ti.static(range(19)):
                out[index] += self.M[index, s] * self.F[i, j, k, s]
        return out

    @ti.func
    def GuoF(self,i,j,k,s,u,f):
       out=0.0
       for l in ti.static(range(19)):
            out += self.w[l]*((self.e_f[l]-u).dot(f)+(self.e_f[l].dot(u)*(self.e_f[l].dot(f))))*self.M[s,l]
       
       return out


    @ti.func
    def meq_vec(self,rho_local,u):
        out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        out[0] = rho_local;             out[3] = u[0];    out[5] = u[1];    out[7] = u[2];
        out[1] = u.dot(u);    out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z;         out[11] = u.y*u.y-u.z*u.z
        out[13] = u.x*u.y;    out[14] = u.y*u.z;                            out[15] = u.x*u.z
        return out

    @ti.func
    def Compute_C(self,i):
        # C = ts.vecFill(3,0.0)
        C = ti.Vector([0.0,0.0,0.0])
        ind_S = 0
        for s in ti.static(range(19)):
            ip = self.periodic_index_for_psi(i+self.e[s])
            if (self.solid[ip] == 0):
                C += 3.0*self.w[s]*self.e_f[s]*self.psi[ip]
            else:
                ind_S = 1
                C += 3.0*self.w[s]*self.e_f[s]*self.psi_solid

        if (abs(self.rho_r[i]-self.rho_b[i]) > 0.9) and (ind_S == 1):
            # C = ts.vecFill(3,0.0)
            C = ti.Vector([0.0,0.0,0.0])
        
        return C

    @ti.func
    def Compute_S_local(self,id):
        sv=0.0; sother=0.0
        if (self.psi[id]>0):
            if (self.psi[id]>0.1):   
                sv=self.wl
            else:
                sv=self.lg0+self.l1*self.psi[id]+self.l2*self.psi[id]*self.psi[id]
        else:
            if (self.psi[id]<-0.1):
                sv=self.wg
            else:
                sv=self.lg0+self.g1*self.psi[id]+self.g2*self.psi[id]*self.psi[id]
        sother = 8.0*(2.0-sv)/(8.0-sv)

        #S = ts.vecFill(19,0.0)
        S = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        S[1]=sv;S[2]=sv;S[4]=sother;S[6]=sother;S[8]=sother;S[9]=sv;
        S[10]=sv;S[11]=sv;S[12]=sv;S[13]=sv;S[14]=sv;S[15]=sv;S[16]=sother;
        S[17]=sother;S[18]=sother;


        return S;

    @ti.kernel
    def colission(self):
        for i,j,k in self.rho:
            #if (solid[i,j,k]==0):
            if (i<self.nx and j<self.ny and k<self.nz and self.solid[i,j,k] == 0):
                uu = self.v[i,j,k].norm_sqr()
                C = self.Compute_C(ti.Vector([i,j,k]))
                cc = C.norm()
                normal = ti.Vector([0.0,0.0,0.0])
                if cc>0 :
                    normal = C/cc

                m_temp = self.multiply_M(i,j,k)
                meq = self.meq_vec(self.rho[i,j,k],self.v[i,j,k])

                meq[1] += self.CapA*cc
                meq[9] += 0.5*self.CapA*cc*(2*normal.x*normal.x-normal.y*normal.y-normal.z*normal.z)
                meq[11] += 0.5*self.CapA*cc*(normal.y*normal.y-normal.z*normal.z)
                meq[13] += 0.5*self.CapA*cc*(normal.x*normal.y)
                meq[14] += 0.5*self.CapA*cc*(normal.y*normal.z)
                meq[15] += 0.5*self.CapA*cc*(normal.x*normal.z)

                S_local = self.Compute_S_local(ti.Vector([i,j,k]))

                #print('~~~',S_local)
                #print('@@@',S_dig)

                #print(i,j,k,solid[i,j,k],m_temp, meq)
                for s in ti.static(range(19)):
                    m_temp[s] -= S_local[s]*(m_temp[s]-meq[s])
                    #Guo force issue.
                    # out = self.GuoF(i,j,k,s,self.v[i,j,k],self.ext_f[None])
                    # print(out)
                    m_temp[s] += (1-0.5*S_local[s])*self.GuoF(i,j,k,s,self.v[i,j,k],self.ext_f[None])
                    #m_temp[s] -= S_dig[s]*(m_temp[s]-meq[s])
                    #m_temp[s] += (1-0.5*S_dig[s])*GuoF(i,j,k,s,v[i,j,k])
                
                # g_r = ts.vecFill(19,0.0)
                # g_b = ts.vecFill(19,0.0)
                g_r = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                g_b = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

                for s in ti.static(range(19)):
                    self.f[i,j,k,s] = 0
                    for l in ti.static(range(19)):
                        self.f[i,j,k,s] += self.inv_M[s,l]*m_temp[l]

                    g_r[s] = self.feq(s,self.rho_r[i,j,k],self.v[i,j,k])
                    g_b[s] = self.feq(s,self.rho_b[i,j,k],self.v[i,j,k])

                if (cc>0):
                    for kk in ti.static([1,3,5,7,9,11,13,15,17]):
                        # ef=ts.vector.dot(e[kk],C)
                        ef=self.e[kk].dot(C)
                        cospsi= g_r[kk] if (g_r[kk]<g_r[kk+1]) else g_r[kk+1]
                        cospsi= cospsi if (cospsi<g_b[kk]) else g_b[kk]
                        cospsi=cospsi if (cospsi<g_b[kk+1]) else g_b[kk+1]
                        cospsi*=ef/cc
    
                        g_r[kk]+=cospsi
                        g_r[kk+1]-=cospsi
                        g_b[kk]-=cospsi
                        g_b[kk+1]+=cospsi
                
                for s in ti.static(range(19)):
                    ip = self.periodic_index(ti.Vector([i,j,k])+self.e[s])
                    if (self.solid[ip]==0):
                        self.rhor[ip] += g_r[s]
                        self.rhob[ip] += g_b[s]
                    else:
                        self.rhor[i,j,k] += g_r[s]
                        self.rhob[i,j,k] += g_b[s]
            
            
            

    @ti.func
    def periodic_index(self,i):
        iout = i
        if i[0]<0:     iout[0] = self.nx-1
        if i[0]>self.nx-1:  iout[0] = 0
        if i[1]<0:     iout[1] = self.ny-1
        if i[1]>self.ny-1:  iout[1] = 0
        if i[2]<0:     iout[2] = self.nz-1
        if i[2]>self.nz-1:  iout[2] = 0

        return iout

    @ti.func
    def periodic_index_for_psi(self,i):
        iout = i
        if i[0]<0:     
            if self.bc_psi_x_left == 0:  
                iout[0] = self.nx-1  
            else: 
                iout[0] = 0
        
        if i[0]>self.nx-1:
            if self.bc_psi_x_right==0:
                iout[0] = 0
            else:
                iout[0] = self.nx-1
        
        if i[1]<0:
            if self.bc_psi_y_left == 0:
                iout[1] = self.ny-1
            else:
                iout[1] = 0
        
        if i[1]>self.ny-1:
            if self.bc_psi_y_right==0:
                iout[1] = 0
            else:
                iout[1] = self.ny-1

        if i[2]<0:
            if self.bc_psi_z_left==0:
                iout[2] = self.nz-1
            else: 
                iout[2] = 0

        if i[2]>self.nz-1:
            if self.bc_psi_z_right==0:
                iout[2] = 0
            else:
                iout[2] = self.nz-1

        return iout

    @ti.kernel
    def streaming1(self):
        for i,j,k in self.rho:
            #if (solid[i,j,k] == 0):
            if (i<self.nx and j<self.ny and k<self.nz and self.solid[i,j,k] == 0):
                ci = ti.Vector([i,j,k])
                for s in ti.static(range(19)):
                    ip = self.periodic_index(ci+self.e[s])
                    if (self.solid[ip]==0):
                        self.F[ip,s] = self.f[ci,s]
                    else:
                        self.F[ci,self.LR[s]] = self.f[ci,s]
                        #print(i, ip, "@@@")

    @ti.kernel
    def Boundary_condition_psi(self):
        if self.bc_psi_x_left == 1:
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[0,j,k]==0):
                    self.psi[0,j,k] = self.psi_x_left
                    self.rho_r[0,j,k] = (self.psi_x_left + 1.0)/2.0
                    self.rho_b[0,j,k] = 1.0 - self.rho_r[0,j,k]

        if self.bc_psi_x_right == 1:
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[self.nx-1,j,k]==0):
                    self.psi[self.nx-1,j,k] = self.psi_x_right
                    self.rho_r[self.nx-1,j,k] = (self.psi_x_right + 1.0)/2.0
                    self.rho_b[self.nx-1,j,k] = 1.0 - self.rho_r[self.nx-1,j,k]

        if self.bc_psi_y_left == 1:
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,0,k]==0):
                    self.psi[i,0,k] = self.psi_y_left
                    self.rho_r[i,0,k] = (self.psi_y_left + 1.0)/2.0
                    self.rho_b[i,0,k] = 1.0 - self.rho_r[i,0,k]
        
        if self.bc_psi_y_right == 1:
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,self.ny-1,k]==0):
                    self.psi[i,self.ny-1,k] = self.psi_y_right
                    self.rho_r[i,self.ny-1,k] = (self.psi_y_right + 1.0)/2.0
                    self.rho_b[i,self.ny-1,k] = 1.0 - self.rho_r[i,self.ny-1,k]

        if self.bc_psi_z_left == 1:
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,0]==0):
                    self.psi[i,j,0] = self.psi_z_left
                    self.rho_r[i,j,0] = (self.psi_z_left + 1.0)/2.0
                    self.rho_b[i,j,0] = 1.0 - self.rho_r[i,j,0]
        
        if self.bc_psi_z_right == 1:
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,self.nz-1]==0):
                    self.psi[i,j,self.nz-1] = self.psi_z_right
                    self.rho_r[i,j,self.nz-1] = (self.psi_z_right + 1.0)/2.0
                    self.rho_b[i,j,self.nz-1] = 1.0 - self.rho_r[i,j,self.nz-1]



    @ti.kernel
    def Boundary_condition(self):
        if ti.static(self.bc_x_left==1):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[1,j,k]>0):
                            self.F[0,j,k,s]=self.feq(s, self.rho_bcxl, self.v[1,j,k])
                        else:
                            self.F[0,j,k,s]=self.feq(s, self.rho_bcxl, self.v[0,j,k])

        if ti.static(self.bc_x_left==2):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        self.F[0,j,k,s]=self.feq(self.LR[s], 1.0, self.bc_vel_x_left[None])-self.F[0,j,k,self.LR[s]]+self.feq(s,1.0,self.bc_vel_x_left[None])  

        if ti.static(self.bc_x_right==1):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[self.nx-1,j,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[self.nx-2,j,k]>0):
                            self.F[self.nx-1,j,k,s]=self.feq(s, self.rho_bcxr, self.v[self.nx-2,j,k])
                        else:
                            self.F[self.nx-1,j,k,s]=self.feq(s, self.rho_bcxr, self.v[self.nx-1,j,k])

        if ti.static(self.bc_x_right==2):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[self.nx-1,j,k]==0):
                    for s in ti.static(range(19)):
                        self.F[self.nx-1,j,k,s]=self.feq(self.LR[s], 1.0, self.bc_vel_x_right[None])-self.F[self.nx-1,j,k,self.LR[s]]+self.feq(s,1.0,self.bc_vel_x_right[None]) 

        
        # Direction Y
        if ti.static(self.bc_y_left==1):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,0,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[i,1,k]>0):
                            self.F[i,0,k,s]=self.feq(s, self.rho_bcyl, self.v[i,1,k])
                        else:
                            self.F[i,0,k,s]=self.feq(s, self.rho_bcyl, self.v[i,0,k])

        if ti.static(self.bc_y_left==2):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,0,k]==0):
                    for s in ti.static(range(19)):
                        self.F[i,0,k,s]=self.feq(self.LR[s], 1.0, self.bc_vel_y_left[None])-self.F[i,0,k,self.LR[s]]+self.feq(s,1.0,self.bc_vel_y_left[None])  

        if ti.static(self.bc_y_right==1):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,self.ny-1,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[i,self.ny-2,k]>0):
                            self.F[i,self.ny-1,k,s]=self.feq(s, self.rho_bcyr, self.v[i,self.ny-2,k])
                        else:
                            self.F[i,self.ny-1,k,s]=self.feq(s, self.rho_bcyr, self.v[i,self.ny-1,k])

        if ti.static(self.bc_y_right==2):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                if (self.solid[i,self.ny-1,k]==0):
                    for s in ti.static(range(19)):
                        self.F[i,self.ny-1,k,s]=self.feq(self.LR[s], 1.0, self.bc_vel_y_right[None])-self.F[i,self.ny-1,k,self.LR[s]]+self.feq(s,1.0,self.bc_vel_y_right[None]) 
        
        # Z direction
        if ti.static(self.bc_z_left==1):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,0]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[i,j,1]>0):
                            self.F[i,j,0,s]=self.feq(s, self.rho_bczl, self.v[i,j,1])
                        else:
                            self.F[i,j,0,s]=self.feq(s, self.rho_bczl, self.v[i,j,0])

        if ti.static(self.bc_z_left==2):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,0]==0):
                    for s in ti.static(range(19)):
                        self.F[i,j,0,s]=self.feq(self.LR[s], 1.0, self.bc_vel_z_left[None])-self.F[i,j,0,self.LR[s]]+self.feq(s,1.0,self.bc_vel_z_left[None])  

        if ti.static(self.bc_z_right==1):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,self.nz-1]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[i,j,self.nz-2]>0):
                            self.F[i,j,self.nz-1,s]=self.feq(s, self.rho_bczr, self.v[i,j,self.nz-2])
                        else:
                            self.F[i,j,self.nz-1,s]=self.feq(s, self.rho_bczr, self.v[i,j,self.nz-1])

        if ti.static(self.bc_z_right==2):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                if (self.solid[i,j,self.nz-1]==0):
                    for s in ti.static(range(19)):
                        self.F[i,j,self.nz-1,s]=self.feq(self.LR[s], 1.0, self.bc_vel_z_right[None])-self.F[i,j,self.nz-1,self.LR[s]]+self.feq(s,1.0,self.bc_vel_z_right[None]) 
        

    @ti.kernel
    def streaming3(self):
        for i,j,k, in self.rho:
            #if (solid[i,j,k] == 0):
            if (i<self.nx and j<self.ny and k<self.nz and self.solid[i,j,k] == 0):
                self.rho[i,j,k] = 0
                self.v[i,j,k] = ti.Vector([0,0,0])

                self.rho_r[i,j,k] = self.rhor[i,j,k]
                self.rho_b[i,j,k] = self.rhob[i,j,k]
                self.rhor[i,j,k] = 0.0; self.rhob[i,j,k] = 0.0

                for s in ti.static(range(19)):
                    self.f[i,j,k,s] = self.F[i,j,k,s]
                    self.rho[i,j,k] += self.f[i,j,k,s]
                    self.v[i,j,k] += self.e_f[s]*self.f[i,j,k,s]
                
                self.v[i,j,k] /= self.rho[i,j,k]
                self.v[i,j,k] += (self.ext_f[None]/2)/self.rho[i,j,k]
                self.psi[i,j,k] = self.rho_r[i,j,k]-self.rho_b[i,j,k]/(self.rho_r[i,j,k] + self.rho_b[i,j,k])

    def get_max_v(self):
        self.max_v[None] = -1e10
        self.cal_max_v()
        return self.max_v[None]
    
    @ti.kernel
    def cal_max_v(self):
        for I in ti.grouped(self.rho):
            ti.atomic_max(self.max_v[None], self.v[I].norm())

    def export_VTK(self, n):
        gridToVTK(
                "./structured_"+str(n),
                self.x,
                self.y,
                self.z,
                #cellData={"pressure": pressure},
                pointData={ "Solid": np.ascontiguousarray(self.solid.to_numpy()),
                            "rho": np.ascontiguousarray(self.rho.to_numpy()[0:self.nx,0:self.ny,0:self.nz]),
                            "phase": np.ascontiguousarray(self.psi.to_numpy()[0:self.nx,0:self.ny,0:self.nz]),
                            "velocity": (np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,0]), np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,1]),np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,2]))
                            }
            )   

    def step(self):
        self.colission()
        self.streaming1()
        self.Boundary_condition()
        #streaming2()
        self.streaming3()
        self.Boundary_condition_psi()
                

# time_init = time.time()
# time_now = time.time()
# time_pre = time.time()
# dt_count = 0               


# solid_np, phase_np = init_geo('./img_ftb131.txt','./phase_ftb131.dat')

# #solid_np = init_geo('./img_ftb131.txt')
# solid.from_numpy(solid_np)
# psi.from_numpy(phase_np)

# static_init()
# init()

# #print(self.,wg, lg0, l1, l2,'~@@@@@~@~@~@~@')

# for iter in range(80000+1):
#     colission()
#     streaminself.()
#     Boundary_condition()
#     #streaming2()
#     streaming3()
#     Boundary_condition_psi()

    
#     if (iter%500==0):
        
#         time_pre = time_now
#         time_now = time.time()
#         diff_time = int(time_now-time_pre)
#         elap_time = int(time_now-time_init)
#         m_diff, s_diff = divmod(diff_time, 60)
#         h_diff, m_diff = divmod(m_diff, 60)
#         m_elap, s_elap = divmod(elap_time, 60)
#         h_elap, m_elap = divmod(m_elap, 60)
        
#         print('----------Time between two outputs is %dh %dm %ds; elapsed time is %dh %dm %ds----------------------' %(h_diff, m_diff, s_diff,h_elap,m_elap,s_elap))
#         print('The %dth iteration, Max Force = %f,  force_scale = %f\n\n ' %(iter, 10.0,  10.0))
        
#         if (iter%10000==0):
#             gridToVTK(
#                 "./structured"+str(iter),
#                 x,
#                 y,
#                 z,
#                 #cellData={"pressure": pressure},
#                 pointData={ "Solid": np.ascontiguousarray(solid.to_numpy()),
#                             "rho": np.ascontiguousarray(rho.to_numpy()[0:nx,0:ny,0:nz]),
#                             "phase": np.ascontiguousarray(psi.to_numpy()[0:nx,0:ny,0:nz]),
#                             "velocity": (np.ascontiguousarray(v.to_numpy()[0:nx,0:ny,0:nz,0]), np.ascontiguousarray(v.to_numpy()[0:nx,0:ny,0:nz,1]),np.ascontiguousarray(v.to_numpy()[0:nx,0:ny,0:nz,2]))
#                             }
#             )   

#ti.print_kernel_profile_info()
#ti.print_profile_info()
