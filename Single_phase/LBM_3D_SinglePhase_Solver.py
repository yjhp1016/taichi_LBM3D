from sympy import inverse_mellin_transform
import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK
import time

ti.init(arch=ti.cuda, dynamic_index=False, kernel_profiler=True, print_ir=False)

@ti.data_oriented
class LB3D_Solver_Single_Phase:
    def __init__(self, nx, ny, nz):

        self.enable_projection = True
        self.nx,self.ny,self.nz = nx,ny,nz
        #nx,ny,nz = 120,120,120
        self.fx,self.fy,self.fz = 0.0e-6,0.0,0.0
        self.niu = 0.16667

        #Boundary condition mode: 0=periodic, 1= fix pressure, 2=fix velocity; boundary pressure value (rho); boundary velocity value for vx,vy,vz
        self.bc_x_left, self.rho_bcxl, self.vx_bcxl, self.vy_bcxl, self.vz_bcxl = 0, 1.0, 0.0e-5, 0.0, 0.0  #Boundary x-axis left side
        self.bc_x_right, self.rho_bcxr, self.vx_bcxr, self.vy_bcxr, self.vz_bcxr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
        self.bc_y_left, self.rho_bcyl, self.vx_bcyl, self.vy_bcyl, self.vz_bcyl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
        self.bc_y_right, self.rho_bcyr, self.vx_bcyr, self.vy_bcyr, self.vz_bcyr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
        self.bc_z_left, self.rho_bczl, self.vx_bczl, self.vy_bczl, self.vz_bczl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
        self.bc_z_right, self.rho_bczr, self.vx_bczr, self.vy_bczr, self.vz_bczr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side


        self.f = ti.Vector.field(19,ti.f32,shape=(nx,ny,nz))
        self.F = ti.Vector.field(19,ti.f32,shape=(nx,ny,nz))
        self.rho = ti.field(ti.f32, shape=(nx,ny,nz))
        self.v = ti.Vector.field(3,ti.f32, shape=(nx,ny,nz))
        self.e = ti.Vector.field(3,ti.i32, shape=(19))
        self.S_dig = ti.Vector.field(19,ti.f32,shape=())
        self.e_f = ti.Vector.field(3,ti.f32, shape=(19))
        self.w = ti.field(ti.f32, shape=(19))
        self.solid = ti.field(ti.i32,shape=(nx,ny,nz))

        self.ext_f = ti.Vector.field(3,ti.f32,shape=())


        self.M = ti.Matrix.field(19, 19, ti.f32, shape=())
        self.inv_M = ti.Matrix.field(19,19,ti.f32, shape=())

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
        
        self.inv_M[None] = ti.Matrix(inv_M_np)

        self.x = np.linspace(0, nx, nx)
        self.y = np.linspace(0, ny, ny)
        self.z = np.linspace(0, nz, nz)
        #X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')


    def init_simulation(self):
        self.bc_vel_x_left = [self.vx_bcxl, self.vy_bcxl, self.vz_bcxl]
        self.bc_vel_x_right = [self.vx_bcxr, self.vy_bcxr, self.vz_bcxr]
        self.bc_vel_y_left = [self.vx_bcyl, self.vy_bcyl, self.vz_bcyl]
        self.bc_vel_y_right = [self.vx_bcyr, self.vy_bcyr, self.vz_bcyr]
        self.bc_vel_z_left = [self.vx_bczl, self.vy_bczl, self.vz_bczl]
        self.bc_vel_z_right = [self.vx_bczr, self.vy_bczr, self.vz_bczr]

        self.tau_f=3.0*self.niu+0.5
        self.s_v=1.0/self.tau_f
        self.s_other=8.0*(2.0-self.s_v)/(8.0-self.s_v)

        self.S_dig[None] = ti.Vector([0,self.s_v,self.s_v,0,self.s_other,0,self.s_other,0,self.s_other, self.s_v, self.s_v,self.s_v,self.s_v,self.s_v,self.s_v,self.s_v,self.s_other,self.s_other,self.s_other])

        self.ext_f[None] = ti.Vector([self.fx,self.fy,self.fz])
        if ((abs(self.fx)>0) or (abs(self.fx)>0) or (abs(self.fx)>0)):
            self.force_flag = 1
        else:
            self.force_flag = 0


        ti.static(self.inv_M)
        ti.static(self.M)
        #ti.static(LR)
        ti.static(self.S_dig)


    @ti.func
    def feq(self, k,rho_local, u):
        eu = self.e[k].dot(u)
        uv = u.dot(u)
        feqout = self.w[k]*rho_local*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
        #print(k, rho_local, self.w[k])
        return feqout

    @ti.kernel
    def init(self):
        for i,j,k in self.rho:
            self.rho[i,j,k] = 1.0
            self.v[i,j,k] = ti.Vector([0,0,0])
            for s in ti.static(range(19)):
                self.f[i,j,k][s] = self.feq(s,1.0,self.v[i,j,k])
                self.F[i,j,k][s] = self.feq(s,1.0,self.v[i,j,k])
                #print(F[i,j,k,s], feq(s,1.0,v[i,j,k]))

   
    def init_geo(self,filename):
        in_dat = np.loadtxt(filename)
        in_dat = np.reshape(in_dat, (self.nx,self.ny,self.nz),order='F')
        self.solid.from_numpy(in_dat)
        

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
    def GuoF(self,i,j,k,s,u):
        out=0.0
        for l in ti.static(range(19)):
            out += self.w[l]*((self.e_f[l]-u).dot(self.ext_f[None])+(self.e_f[l].dot(u)*(self.e_f[l].dot(self.ext_f[None]))))*self.M[None][s,l]
        
        return out


    @ti.func
    def meq_vec(self, rho_local,u):
        out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        out[0] = rho_local;             out[3] = u[0];    out[5] = u[1];    out[7] = u[2];
        out[1] = u.dot(u);    out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z;         out[11] = u.y*u.y-u.z*u.z
        out[13] = u.x*u.y;    out[14] = u.y*u.z;                            out[15] = u.x*u.z
        return out



    @ti.kernel
    def colission(self):
        for i,j,k in self.rho:
            if (self.solid[i,j,k] == 0):
                m_temp = self.M[None]@self.F[i,j,k]
                meq = self.meq_vec(self.rho[i,j,k],self.v[i,j,k])
                m_temp -= self.S_dig[None]*(m_temp-meq)
                if (ti.static(self.force_flag==1)):
                    for s in ti.static(range(19)):
                    #    m_temp[s] -= S_dig[s]*(m_temp[s]-meq[s])
                        m_temp[s] += (1-0.5*self.S_dig[None][s])*self.GuoF(i,j,k,s,self.v[i,j,k])
                
                self.f[i,j,k] = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                self.f[i,j,k] += self.inv_M[None]@m_temp


           
           

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

    @ti.kernel
    def streaming1(self):
        for i in ti.grouped(self.rho):
            if (self.solid[i] == 0):
                for s in ti.static(range(19)):
                    ip = self.periodic_index(i+self.e[s])
                    if (self.solid[ip]==0):
                        self.F[ip][s] = self.f[i][s]
                    else:
                        self.F[i][self.LR[s]] = self.f[i][s]
                        #print(i, ip, "@@@")


    @ti.kernel
    def Boundary_condition(self):
        if ti.static(self.bc_x_left==1):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[1,j,k]>0):
                            self.F[0,j,k][s]=self.feq(s, self.rho_bcxl, self.v[1,j,k])
                        else:
                            self.F[0,j,k][s]=self.feq(s, self.rho_bcxl, self.v[0,j,k])

        if ti.static(self.bc_x_left==2):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[0,j,k]==0):
                    for s in ti.static(range(19)):
                        #F[0,j,k][s]=feq(LR[s], 1.0, bc_vel_x_left[None])-F[0,j,k,LR[s]]+feq(s,1.0,bc_vel_x_left[None])  #!!!!!!change velocity in feq into vector
                        self.F[0,j,k][s]=self.feq(s,1.0,ti.Vector(self.bc_vel_x_left))

        if ti.static(self.bc_x_right==1):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[nx-1,j,k]==0):
                    for s in ti.static(range(19)):
                        if (self.solid[self.nx-2,j,k]>0):
                            self.F[self.nx-1,j,k][s]=self.feq(s, self.rho_bcxr, self.v[self.nx-2,j,k])
                        else:
                            self.F[self.nx-1,j,k][s]=self.feq(s, self.rho_bcxr, self.v[self.nx-1,j,k])

        if ti.static(self.bc_x_right==2):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                if (self.solid[self.nx-1,j,k]==0):
                    for s in ti.static(range(19)):
                        #F[nx-1,j,k][s]=feq(LR[s], 1.0, bc_vel_x_right[None])-F[nx-1,j,k,LR[s]]+feq(s,1.0,bc_vel_x_right[None])  #!!!!!!change velocity in feq into vector
                        self.F[self.nx-1,j,k][s]=self.feq(s,1.0,ti.Vector(self.bc_vel_x_right))


    @ti.kernel
    def streaming3(self):
        for i in ti.grouped(self.rho):
            if (self.solid[i]==0):
                self.rho[i] = 0
                self.v[i] = ti.Vector([0,0,0])
                self.f[i] = self.F[i]
                self.rho[i] += self.f[i].sum()

                for s in ti.static(range(19)):
                    self.v[i] += self.e_f[s]*self.f[i][s]
                
                self.v[i] /= self.rho[i]
                self.v[i] += (self.ext_f[None]/2)/self.rho[i]
                
            else:
                self.rho[i] = 1.0
                self.v[i] = ti.Vector([0,0,0])
    
    def set_bc_vel_x1(self, vel):
        self.bc_x_right = 2
        self.vx_bcxr = vel[0]; self.vy_bcxr = vel[1]; self.vz_bcxr = vel[2];

    def set_viscosity(self,niu):
        self.niu = niu

    def export_VTK(self, n):
        gridToVTK(
                "./LB_SingelPhase_"+str(n),
                self.x,
                self.y,
                self.z,
                #cellData={"pressure": pressure},
                pointData={ "Solid": np.ascontiguousarray(self.solid.to_numpy()),
                            "rho": np.ascontiguousarray(self.rho.to_numpy()),
                            "velocity": (np.ascontiguousarray(self.v.to_numpy()[:,:,:,0]), np.ascontiguousarray(self.v.to_numpy()[:,:,:,1]),np.ascontiguousarray(self.v.to_numpy()[:,:,:,2]))
                            }
            )   

    def step(self):
        lb3d.colission()
        lb3d.streaming1()
        lb3d.Boundary_condition()
        lb3d.streaming3()






time_init = time.time()
time_now = time.time()
time_pre = time.time()
dt_count = 0               


lb3d = LB3D_Solver_Single_Phase(50,50,50)

lb3d.init_geo('./geo_cavity.dat')
lb3d.set_bc_vel_x1([0.0,0.0,0.1])
lb3d.init_simulation()
lb3d.static_init()
lb3d.init()

for iter in range(50000+1):
    lb3d.step()

    if (iter%1000==0):
        
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
            lb3d.export_VTK(iter)
            


#ti.profiler.print_scoped_profiler_info()
#ti.print_profile_info()
#ti.profiler.print_kernel_profiler_info()  # default mode: 'count'

#ti.profiler.print_kernel_profiler_info('trace')
#ti.profiler.clear_kernel_profiler_info()  # clear all records