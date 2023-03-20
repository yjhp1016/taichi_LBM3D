#from sympy import inverse_mellin_transform
import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK
import time

ti.init(arch=ti.gpu, dynamic_index=False, kernel_profiler=True, print_ir=False)

enable_projection = True
nx,ny,nz = 50,50,50
#nx,ny,nz = 120,120,120
fx,fy,fz = 0.0e-6,0.0,0.0
niu = 0.16667

#Boundary condition mode: 0=periodic, 1= fix pressure, 2=fix velocity; boundary pressure value (rho); boundary velocity value for vx,vy,vz
bc_x_left, rho_bcxl, vx_bcxl, vy_bcxl, vz_bcxl = 0, 1.0, 0.0e-5, 0.0, 0.0  #Boundary x-axis left side
bc_x_right, rho_bcxr, vx_bcxr, vy_bcxr, vz_bcxr = 2, 1.0, 0.0, 0.0, 0.1  #Boundary x-axis left side
bc_y_left, rho_bcyl, vx_bcyl, vy_bcyl, vz_bcyl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
bc_y_right, rho_bcyr, vx_bcyr, vy_bcyr, vz_bcyr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
bc_z_left, rho_bczl, vx_bczl, vy_bczl, vz_bczl = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side
bc_z_right, rho_bczr, vx_bczr, vy_bczr, vz_bczr = 0, 1.0, 0.0, 0.0, 0.0  #Boundary x-axis left side


f = ti.Vector.field(19,ti.f32,shape=(nx,ny,nz))
F = ti.Vector.field(19,ti.f32,shape=(nx,ny,nz))
rho = ti.field(ti.f32, shape=(nx,ny,nz))
v = ti.Vector.field(3,ti.f32, shape=(nx,ny,nz))
e = ti.Vector.field(3,ti.i32, shape=(19))
S_dig = ti.Vector.field(19,ti.f32,shape=())
e_f = ti.Vector.field(3,ti.f32, shape=(19))
w = ti.field(ti.f32, shape=(19))
solid = ti.field(ti.i32,shape=(nx,ny,nz))

ext_f = ti.Vector.field(3,ti.f32,shape=())


M = ti.Matrix.field(19, 19, ti.f32, shape=())
inv_M = ti.Matrix.field(19,19,ti.f32, shape=())

bc_vel_x_left = [vx_bcxl, vy_bcxl, vz_bcxl]
bc_vel_x_right = [vx_bcxr, vy_bcxr, vz_bcxr]
bc_vel_y_left = [vx_bcyl, vy_bcyl, vz_bcyl]
bc_vel_y_right = [vx_bcyr, vy_bcyr, vz_bcyr]
bc_vel_z_left = [vx_bczl, vy_bczl, vz_bczl]
bc_vel_z_right = [vx_bczr, vy_bczr, vz_bczr]

tau_f=3.0*niu+0.5
s_v=1.0/tau_f
s_other=8.0*(2.0-s_v)/(8.0-s_v)

S_np = np.zeros((19,19))
S_np[0,0]=0;        S_np[1,1]=s_v;          S_np[2,2]=s_v;          S_np[3,3]=0;        S_np[4,4]=s_other;      S_np[5,5]=0;
S_np[6,6]=s_other;  S_np[7,7]=0;            S_np[8,8]=s_other;      S_np[9,9]=s_v;      S_np[10,10]=s_v;        S_np[11,11]=s_v;
S_np[12,12]=s_v;    S_np[13,13]=s_v;        S_np[14,14]=s_v;        S_np[15,15]=s_v;    S_np[16,16]=s_other;    S_np[17,17]=s_other;
S_np[18,18]=s_other

S_dig_np = np.array([0,s_v,s_v,0,s_other,0,s_other,0,s_other, s_v, s_v,s_v,s_v,s_v,s_v,s_v,s_other,s_other,s_other])

LR = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17]

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


M[None] = ti.Matrix(M_np)
inv_M[None] = ti.Matrix(inv_M_np)
S_dig[None] = ti.Vector(S_dig_np)


ext_f[None] = ti.Vector([fx,fy,fz])
if ((abs(fx)>0) or (abs(fx)>0) or (abs(fx)>0)):
    force_flag = 1
else:
    force_flag = 0


ti.static(inv_M)
ti.static(M)
#ti.static(LR)
ti.static(S_dig)


x = np.linspace(0, nx, nx)
y = np.linspace(0, ny, ny)
z = np.linspace(0, nz, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


@ti.func
def feq(k,rho_local, u):
    eu = e[k].dot(u)
    uv = u.dot(u)
    feqout = w[k]*rho_local*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
    #print(k, rho_local, w[k])
    return feqout

@ti.kernel
def init():
    for i,j,k in rho:
        rho[i,j,k] = 1.0
        v[i,j,k] = ti.Vector([0,0,0])
        for s in ti.static(range(19)):
            f[i,j,k][s] = feq(s,1.0,v[i,j,k])
            F[i,j,k][s] = feq(s,1.0,v[i,j,k])
            #print(F[i,j,k,s], feq(s,1.0,v[i,j,k]))

    M[None] = ti.Matrix(M_np)
    inv_M[None] = ti.Matrix(inv_M_np)
    S_dig[None] = ti.Vector(S_dig_np)
   

def init_geo(filename):
    in_dat = np.loadtxt(filename)
    in_dat = np.reshape(in_dat, (nx,ny,nz),order='F')
    return in_dat

@ti.kernel
def static_init():
  if ti.static(enable_projection): # No runtime overhead
    e[0] = ti.Vector([0,0,0])
    e[1] = ti.Vector([1,0,0]); e[2] = ti.Vector([-1,0,0]); e[3] = ti.Vector([0,1,0]); e[4] = ti.Vector([0,-1,0]);e[5] = ti.Vector([0,0,1]); e[6] = ti.Vector([0,0,-1])
    e[7] = ti.Vector([1,1,0]); e[8] = ti.Vector([-1,-1,0]); e[9] = ti.Vector([1,-1,0]); e[10] = ti.Vector([-1,1,0])
    e[11] = ti.Vector([1,0,1]); e[12] = ti.Vector([-1,0,-1]); e[13] = ti.Vector([1,0,-1]); e[14] = ti.Vector([-1,0,1])
    e[15] = ti.Vector([0,1,1]); e[16] = ti.Vector([0,-1,-1]); e[17] = ti.Vector([0,1,-1]); e[18] = ti.Vector([0,-1,1])

    e_f[0] = ti.Vector([0,0,0])
    e_f[1] = ti.Vector([1,0,0]); e_f[2] = ti.Vector([-1,0,0]); e_f[3] = ti.Vector([0,1,0]); e_f[4] = ti.Vector([0,-1,0]);e_f[5] = ti.Vector([0,0,1]); e_f[6] = ti.Vector([0,0,-1])
    e_f[7] = ti.Vector([1,1,0]); e_f[8] = ti.Vector([-1,-1,0]); e_f[9] = ti.Vector([1,-1,0]); e_f[10] = ti.Vector([-1,1,0])
    e_f[11] = ti.Vector([1,0,1]); e_f[12] = ti.Vector([-1,0,-1]); e_f[13] = ti.Vector([1,0,-1]); e_f[14] = ti.Vector([-1,0,1])
    e_f[15] = ti.Vector([0,1,1]); e_f[16] = ti.Vector([0,-1,-1]); e_f[17] = ti.Vector([0,1,-1]); e_f[18] = ti.Vector([0,-1,1])

    w[0] = 1.0/3.0; w[1] = 1.0/18.0; w[2] = 1.0/18.0; w[3] = 1.0/18.0; w[4] = 1.0/18.0; w[5] = 1.0/18.0; w[6] = 1.0/18.0; 
    w[7] = 1.0/36.0; w[8] = 1.0/36.0; w[9] = 1.0/36.0; w[10] = 1.0/36.0; w[11] = 1.0/36.0; w[12] = 1.0/36.0; 
    w[13] = 1.0/36.0; w[14] = 1.0/36.0; w[15] = 1.0/36.0; w[16] = 1.0/36.0; w[17] = 1.0/36.0; w[18] = 1.0/36.0;


@ti.func
def GuoF(i,j,k,s,u):
    out=0.0
    for l in ti.static(range(19)):
        out += w[l]*((e_f[l]-u).dot(ext_f[None])+(e_f[l].dot(u)*(e_f[l].dot(ext_f[None]))))*M[None][s,l]
    
    return out


@ti.func
def meq_vec(rho_local,u):
    out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    out[0] = rho_local;             out[3] = u[0];    out[5] = u[1];    out[7] = u[2];
    out[1] = u.dot(u);    out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z;         out[11] = u.y*u.y-u.z*u.z
    out[13] = u.x*u.y;    out[14] = u.y*u.z;                            out[15] = u.x*u.z
    return out



@ti.kernel
def colission():
    for i,j,k in rho:
        if (solid[i,j,k] == 0):
            m_temp = M[None]@F[i,j,k]
            meq = meq_vec(rho[i,j,k],v[i,j,k])
            m_temp -= S_dig[None]*(m_temp-meq)
            if (ti.static(force_flag==1)):
                for s in ti.static(range(19)):
                #    m_temp[s] -= S_dig[s]*(m_temp[s]-meq[s])
                    m_temp[s] += (1-0.5*S_dig[None][s])*GuoF(i,j,k,s,v[i,j,k])
            
            f[i,j,k] = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            f[i,j,k] += inv_M[None]@m_temp


           
           

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

@ti.kernel
def streaming1():
    for i in ti.grouped(rho):
        if (solid[i] == 0):
            for s in ti.static(range(19)):
                ip = periodic_index(i+e[s])
                if (solid[ip]==0):
                    F[ip][s] = f[i][s]
                else:
                    F[i][LR[s]] = f[i][s]
                    #print(i, ip, "@@@")


@ti.kernel
def Boundary_condition():
    if ti.static(bc_x_left==1):
        for j,k in ti.ndrange((0,ny),(0,nz)):
            if (solid[0,j,k]==0):
                for s in ti.static(range(19)):
                    if (solid[1,j,k]>0):
                        F[0,j,k][s]=feq(s, rho_bcxl, v[1,j,k])
                    else:
                        F[0,j,k][s]=feq(s, rho_bcxl, v[0,j,k])

    if ti.static(bc_x_left==2):
        for j,k in ti.ndrange((0,ny),(0,nz)):
            if (solid[0,j,k]==0):
                for s in ti.static(range(19)):
                    #F[0,j,k][s]=feq(LR[s], 1.0, bc_vel_x_left[None])-F[0,j,k,LR[s]]+feq(s,1.0,bc_vel_x_left[None])  #!!!!!!change velocity in feq into vector
                    F[0,j,k][s]=feq(s,1.0,ti.Vector(bc_vel_x_left))

    if ti.static(bc_x_right==1):
        for j,k in ti.ndrange((0,ny),(0,nz)):
            if (solid[nx-1,j,k]==0):
                for s in ti.static(range(19)):
                    if (solid[nx-2,j,k]>0):
                        F[nx-1,j,k][s]=feq(s, rho_bcxr, v[nx-2,j,k])
                    else:
                        F[nx-1,j,k][s]=feq(s, rho_bcxr, v[nx-1,j,k])

    if ti.static(bc_x_right==2):
        for j,k in ti.ndrange((0,ny),(0,nz)):
            if (solid[nx-1,j,k]==0):
                for s in ti.static(range(19)):
                    #F[nx-1,j,k][s]=feq(LR[s], 1.0, bc_vel_x_right[None])-F[nx-1,j,k,LR[s]]+feq(s,1.0,bc_vel_x_right[None])  #!!!!!!change velocity in feq into vector
                    F[nx-1,j,k][s]=feq(s,1.0,ti.Vector(bc_vel_x_right))


@ti.kernel
def streaming3():
    for i in ti.grouped(rho):
        if (solid[i]==0):
            rho[i] = 0
            v[i] = ti.Vector([0,0,0])
            f[i] = F[i]
            rho[i] += f[i].sum()

            for s in ti.static(range(19)):
                v[i] += e_f[s]*f[i][s]
            
            v[i] /= rho[i]
            v[i] += (ext_f[None]/2)/rho[i]
            
        else:
            rho[i] = 1.0
            v[i] = ti.Vector([0,0,0])
        

time_init = time.time()
time_now = time.time()
time_pre = time.time()
dt_count = 0               

#solid_np = init_geo('./BC.dat')
solid_np = init_geo('./geo_cavity.dat')
solid.from_numpy(solid_np)

static_init()
init()
print(bc_vel_x_right)

for iter in range(50000+1):
    colission()
    streaming1()
    Boundary_condition()
    #streaming2()
    streaming3()

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
            gridToVTK(
                "./structured"+str(iter),
                x,
                y,
                z,
                #cellData={"pressure": pressure},
                pointData={ "Solid": np.ascontiguousarray(solid.to_numpy()),
                            "rho": np.ascontiguousarray(rho.to_numpy()),
                            "velocity": (np.ascontiguousarray(v.to_numpy()[:,:,:,0]), np.ascontiguousarray(v.to_numpy()[:,:,:,1]),np.ascontiguousarray(v.to_numpy()[:,:,:,2]))
                            }
            )   
#ti.profiler.print_scoped_profiler_info()
#ti.print_profile_info()
#ti.profiler.print_kernel_profiler_info()  # default mode: 'count'

#ti.profiler.print_kernel_profiler_info('trace')
#ti.profiler.clear_kernel_profiler_info()  # clear all records