LBM_3D_SinglePhase_Solute_Solver
======================================

This file is the solver for solute transportation

First import the certain package and define the class of ``LB3D_Solver_Single_Phase_Solute`` which inheritant from 
``LB3D_Solver_Single_Phase_Solute``

.. code-block:: python

    from sympy import inverse_mellin_transform
    import taichi as ti
    import numpy as np
    from pyevtk.hl import gridToVTK
    import time

    #ti.init(arch=ti.cpu, dynamic_index=False, kernel_profiler=False, print_ir=False)
    import LBM_3D_SinglePhase_Solver as lb3d

    @ti.data_oriented
    class LB3D_Solver_Single_Phase_Solute(lb3d.LB3D_Solver_Single_Phase):
        def __init__(self, nx, ny, nz):
            super(LB3D_Solver_Single_Phase_Solute, self).__init__(nx, ny, nz, sparse_storage = False)
            #define solute boundary condition 
            self.solute_bc_x_left, self.solute_bcxl = 0, 0.0
            self.solute_bc_x_right, self.solute_bcxr = 0, 0.0
            self.solute_bc_y_left, self.solute_bcyl = 0, 0.0
            self.solute_bc_y_right, self.solute_bcyr = 0, 0.0
            self.solute_bc_z_left, self.solute_bczl = 0, 0.0
            self.solute_bc_z_right, self.solute_bczr = 0, 0.0

            #define parameters for bouyancy force
            self.buoyancy_parameter = 20.0   #Buoyancy Parameter (0= no buoyancy)
            self.ref_T = 20.0              #reference_psi F=/rho*g+Bouyancy*(/psi-reference_psi)*g)
            #define gravity
            self.gravity = 5e-7
            
            #define concentration distribution function
            self.fg = ti.Vector.field(19,ti.f32,shape=(nx,ny,nz))
            #define another concentration distribution function
            self.Fg = ti.Vector.field(19,ti.f32,shape=(nx,ny,nz))
            #define external force
            self.forcexyz = ti.Vector.field(3,ti.f32,shape=(nx,ny,nz))
            #define entropy
            self.rho_H = ti.field(ti.f32, shape=(nx,ny,nz))
            #define temperature
            self.rho_T = ti.field(ti.f32, shape=(nx,ny,nz))
            #define liquid volumn fraction
            self.rho_fl = ti.field(ti.f32, shape=(nx,ny,nz))

            #define specific heat of liquid
            self.Cp_l= 1.0
            #define specific heat of solid 
            self.Cp_s = 1.0
            #define latent heat
            self.Lt = 1.0
            #define solid temperature
            self.T_s = -10.0
            #define liquid temperature
            self.T_l = -10.0
            #define viscosity of solid 
            self.niu_s = 0.002
            #define viscosity of liquid
            self.niu_l = 0.002

            #define energy of solid 
            self.H_s = None
            #define energy of liquid
            self.H_l = None

            #define rock thermal diffusivity
            self.niu_solid = 0.001
            #define specific heat of rock
            self.Cp_solid = 1.0

An then it sets these parameters with functions 

.. code-block:: python

    #set gravity
    def set_gravity(self, gravity):
    self.gravity = gravity
    #set buoyancy force parameter
    def set_buoyancy_parameter(self, buoyancy_param):
        self.buoyancy_parameter = buoyancy_param
    #set reference temperature
    def set_ref_T(self, ref_t):
        self.ref_T = ref_t
    #set specific heat of solid
    def set_specific_heat_solid(self, cps):
        self.Cp_s = cps
    #set specfic heat of liquid
    def set_specific_heat_liquid(self, cpl):
        self.Cp_l = cpl
    #set specfic heat of rock
    def set_specific_heat_rock(self, cprock):
        self.Cp_solid = cprock
    #set latent heat
    def set_latent_heat(self, ltheat):
        self.Lt = ltheat
    #set solidus temperature
    def set_solidus_temperature(self, ts):
        self.T_s = ts
    #set liquidus temperature
    def set_liquidus_temperature(self, tl):
        self.T_l = tl
    #set solid thermal diffusivity
    def set_solid_thermal_diffusivity(self, nius):
        self.niu_s = nius
    #set liquid thermal diffusivity
    def set_liquid_thermal_diffusivity(self, niul):
        self.niu_l = niul
    #set rock thermal diffusivity 
    def set_rock_thermal_diffusivity(self, niurock):
        self.niu_solid = niurock
    #set adiabatic boundary on x-left
    def set_bc_adiabatic_x_left(self, bc_ad):
        if (bc_ad==True):
            self.solute_bc_x_left = 2
    #set adiabatic boundary on x-right
    def set_bc_adiabatic_x_right(self, bc_ad):
        if (bc_ad==True):
            self.solute_bc_x_right = 2
    #set adiabatic boundary on y-left
    def set_bc_adiabatic_y_left(self, bc_ad):
        if (bc_ad==True):
            self.solute_bc_y_left = 2
    #set adiabatic boundary on y-right
    def set_bc_adiabatic_y_right(self, bc_ad):
        if (bc_ad==True):
            self.solute_bc_y_right = 2
    #set adiabatic boundary on z-left
    def set_bc_adiabatic_z_left(self, bc_ad):
        if (bc_ad==True):
            self.solute_bc_z_left = 2
    #set adiabatic boundary on z-right
    def set_bc_adiabatic_z_right(self, bc_ad):
        if (bc_ad==True):
            self.solute_bc_z_right = 2
    #set constant temperature on x-left
    def set_bc_constant_temperature_x_left(self,xl):
        self.solute_bc_x_left = 1
        self.solute_bcxl = xl
    #set constant temperature on x-right
    def set_bc_constant_temperature_x_right(self,xr):
        self.solute_bc_x_right = 1
        self.solute_bcxr = xr
    #set constant temperature on y-left
    def set_bc_constant_temperature_y_left(self,yl):
        self.solute_bc_y_left = 1
        self.solute_bcyl = yl
    #set constant temperature on y-right
    def set_bc_constant_temperature_y_right(self,yr):
        self.solute_bc_y_right = 1
        self.solute_bcyr = yr
    #set constant temperature on z-left
    def set_bc_constant_temperature_z_left(self,zl):
        self.solute_bc_z_left = 1
        self.solute_bczl = zl
    #set constant temperature on z-right
    def set_bc_constant_temperature_z_right(self,zr):
        self.solute_bc_y_right = 1
        self.solute_bczr = zr

    # update energy of solid and liquid
    def update_H_sl(self):
        #energy of solid 
        self.H_s = self.Cp_s*self.T_s
        #energy of liquid 
        self.H_l = self.H_s+self.Lt
        print('H_s',self.H_s)
        print('H_l',self.H_l)
    
Then it initialize some variable or function

.. code-block:: python

    #intialize the energy
    @ti.kernel
    def init_H(self):
        for I in ti.grouped(self.rho_T):
            #calculate the energy, convert_T_H() define later
            self.rho_H[I] = self.convert_T_H(self.rho_T[I])
    
    #intialize the density distribiution function for solute concentration 
    @ti.kernel
    def init_fg(self):
        for I in ti.grouped(self.fg):
            #calculate the overall specific heat
            Cp = self.rho_fl[I]*self.Cp_l + (1-self.rho_fl[I])*self.Cp_s
            #intialize the density distribiution function for solute concentration equals equilibrium density distribiution function for solute concentration
            for s in ti.static(range(19)):
                self.fg[I][s] = self.g_feq(s,self.rho_T[I],self.rho_H[I], Cp, self.v[I])
                self.Fg[I][s] = self.fg[I][s]
    
    #intialize the volumn fraction of liquid 
    @ti.kernel
    def init_fl(self):
        for I in ti.grouped(self.rho_T):
            #convert_T_fl define later
            self.rho_fl[I] = self.convert_T_fl(self.rho_T[I])

``g_feq(self, k,local_T,local_H, Cp, u)`` calculate the equilibrium density distribiution function for thermal energy

.. code-block:: python

    @ti.func
    def g_feq(self, k,local_T,local_H, Cp, u):
        eu = self.e[k].dot(u)
        uv = u.dot(u)
        feqout = 0.0
        #calculating the zero-velocity equilibrium thermal distribution function 
        if (k==0):
            feqout = local_H-Cp*local_T+self.w[k]*Cp*local_T*(1-1.5*uv)
        else:
        #calculating other directions equilibrium thermal distribution function 
            feqout = self.w[k]*Cp*local_T*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
        #print(k, self.w[k], feqout, Cp, local_T)
        return feqout

``cal_local_force(i, j, k)`` calculates buoyancy force

.. code-block:: python

        #density is the function of temperture delat(rho)=-rho*beta*delta(T)
        @ti.func
        def cal_local_force(self, i, j, k):
            f = ti.Vector([self.fx, self.fy, self.fz])
            f[1] += self.gravity*self.buoyancy_parameter*(self.rho_T[i,j,k]-self.ref_T)
            #f= delta(rho)*delta(v)*g
            f *= self.rho_fl[i,j,k]
            return f

``collision_g()`` defines the the collision of thermal distribution function

.. code-block:: python 

    @ti.kernel
    def colission_g(self):
        for I in ti.grouped(self.rho_T):
            #overall relaxation time
            tau_s = 3*(self.niu_s*(1.0-self.rho_fl[I])+self.niu_l*self.rho_fl[I])+0.5
            #overall specific heat
            Cp = self.rho_fl[I]*self.Cp_l + (1-self.rho_fl[I])*self.Cp_s

            #ROCK overall relaxation time and specific heat
            if (self.solid[I] >0):
                tau_s = 3.0*self.niu_solid+0.5
                Cp = self.Cp_solid
    
            #f=f-1/tau*(f-feq)
            for s in ti.static(range(19)):
                tmp_fg = -1.0/tau_s*(self.fg[I][s]-self.g_feq(s,self.rho_T[I],self.rho_H[I], Cp, self.v[I]))
                #print(self.fg[I][s],tmp_fg,I,s,self.rho_H[I],self.g_feq(s,self.rho_T[I],self.rho_H[I], Cp, self.v[I]))
                self.fg[I][s] += tmp_fg
            
``collision()`` defines the the collision of density distribution function

.. code-block:: python 

    @ti.kernel
    def colission(self):
        for i,j,k in self.rho:
            #if (self.solid[i,j,k] == 0):
            m_temp = self.M[None]@self.F[i,j,k]
            meq = self.meq_vec(self.rho[i,j,k],self.v[i,j,k])
            m_temp -= self.S_dig[None]*(m_temp-meq)
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
            
            self.f[i,j,k] = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            #calculate the denisty distribution function in momentum space here
            self.f[i,j,k] += self.inv_M[None]@m_temp
            #calculate the fluid density distribution function here
            for s in ti.static(range(19)):
                self.f[i,j,k][s] = self.f[i,j,k][s]*(self.rho_fl[i,j,k]) + self.w[s]*(1.0-self.rho_fl[i,j,k])

``streaming1()`` and ``streaming1_g()`` defines the fluid denisty distribiution function and 
thermal density distribiution function

.. code-block:: python 

    @ti.kernel
    def streaming1(self):
        for i in ti.grouped(self.rho):
            #if (self.solid[i] == 0):
            for s in ti.static(range(19)):
                ip = self.periodic_index(i+self.e[s])
                self.F[ip][s] = self.f[i][s]
                
    @ti.kernel
    def streaming1_g(self):
        for i in ti.grouped(self.rho_T):
            for s in ti.static(range(19)):
                ip = self.periodic_index(i+self.e[s])
                self.Fg[ip][s] = self.fg[i][s]

this 

.. code-block:: python

    @ti.kernel
    def BC_concentration(self):
        #constant temperature boundary condition
        if ti.static(self.solute_bc_x_left==1):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                local_T = self.solute_bcxl
                local_H = self.convert_T_H(local_T)
                Cp = self.rho_fl[0,j,k]*self.Cp_l + (1-self.rho_fl[0,j,k])*self.Cp_s
                #the boundary's thermal distribution function equals the equilibrium thermal distribution function on the boundary
                for s in ti.static(range(19)):
                    self.fg[0,j,k][s] = self.g_feq(s,local_T, local_H, Cp, self.v[0,j,k])
                    self.Fg[0,j,k][s] = self.fg[0,j,k][s]
        #adiabatic boundary condition
        elif ti.static(self.solute_bc_x_left==2):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                for s in ti.static(range(19)):
                #there is no thermal transfer between the boundaty and neighbouring cell
                    self.fg[0,j,k][s] = self.fg[1,j,k][s]
                    self.Fg[0,j,k][s] = self.fg[1,j,k][s]

        #x-right
        if ti.static(self.solute_bc_x_right==1):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                local_T = self.solute_bcxr
                local_H = self.convert_T_H(local_T)
                Cp = self.rho_fl[self.nx-1,j,k]*self.Cp_l + (1-self.rho_fl[self.nx-1,j,k])*self.Cp_s

                for s in ti.static(range(19)):
                    self.fg[self.nx-1,j,k][s] = self.g_feq(s,local_T, local_H, Cp, self.v[self.nx-1,j,k])
                    self.Fg[self.nx-1,j,k][s]= self.fg[self.nx-1,j,k][s]
        elif ti.static(self.solute_bc_x_right==2):
            for j,k in ti.ndrange((0,self.ny),(0,self.nz)):
                for s in ti.static(range(19)):
                    self.fg[self.nx-1,j,k][s] = self.fg[self.nx-2,j,k][s]
                    self.Fg[self.nx-1,j,k][s] = self.fg[self.nx-2,j,k][s]

        #y-left
        if ti.static(self.solute_bc_y_left==1):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                local_T = self.solute_bcyl
                local_H = self.convert_T_H(local_T)
                Cp = self.rho_fl[i,0,k]*self.Cp_l + (1-self.rho_fl[i,0,k])*self.Cp_s

                for s in ti.static(range(19)):
                    self.fg[i,0,k][s] = self.g_feq(s,local_T, local_H, Cp, self.v[i,0,k])
                    self.Fg[i,0,k][s] = self.fg[i,0,k][s]
        elif ti.static(self.solute_bc_y_left==2):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                for s in ti.static(range(19)):
                    self.fg[i,0,k][s] = self.fg[i,1,k][s]
                    self.Fg[i,0,k][s] = self.fg[i,1,k][s]

        #y-right
        if ti.static(self.solute_bc_y_right==1):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                local_T = self.solute_bcyr
                local_H = self.convert_T_H(local_T)
                Cp = self.rho_fl[i,self.ny-1,k]*self.Cp_l + (1-self.rho_fl[i,self.ny-1,k])*self.Cp_s

                for s in ti.static(range(19)):
                    self.fg[i,self.ny-1,k][s] = self.g_feq(s,local_T, local_H, Cp, self.v[i,self.ny-1,k])
                    self.Fg[i,self.ny-1,k][s] = self.fg[i,self.ny-1,k][s]
        elif ti.static(self.solute_bc_y_right==2):
            for i,k in ti.ndrange((0,self.nx),(0,self.nz)):
                for s in ti.static(range(19)):
                    self.fg[i,self.ny-1,k][s] = self.fg[i,self.ny-2,k][s]
                    self.Fg[i,self.ny-1,k][s] = self.fg[i,self.ny-2,k][s]

        #z-left
        if ti.static(self.solute_bc_z_left==1):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                local_T = self.solute_bczl
                local_H = self.convert_T_H(local_T)
                Cp = self.rho_fl[i,j,0]*self.Cp_l + (1-self.rho_fl[i,j,0])*self.Cp_s

                for s in ti.static(range(19)):
                    self.fg[i,j,0][s] = self.g_feq(s,local_T, local_H, Cp, self.v[i,j,0])
                    self.Fg[i,j,0][s] = self.fg[i,j,0][s]
        elif ti.static(self.solute_bc_z_left==1):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                for s in ti.static(range(19)):
                    self.fg[i,j,0][s] = self.fg[i,j,1][s]
                    self.Fg[i,j,0][s] = self.fg[i,j,1][s]

        #z-right
        if ti.static(self.solute_bc_z_right==1):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                local_T = self.solute_bczr
                local_H = self.convert_T_H(local_T)
                Cp = self.rho_fl[i,j,self.nz-1]*self.Cp_l + (1-self.rho_fl[i,j,self.nz-1])*self.Cp_s

                for s in ti.static(range(19)):
                    self.fg[i,j,self.nz-1][s] = self.g_feq(s,local_T, local_H, Cp, self.v[i,j,self.nz-1])
                    self.Fg[i,j,self.nz-1][s] = self.fg[i,j,self.nz-1][s]
        elif ti.static(self.solute_bc_z_right==1):
            for i,j in ti.ndrange((0,self.nx),(0,self.ny)):
                for s in ti.static(range(19)):
                    self.fg[i,j,self.nz-1][s] = self.fg[i,j,self.nz-2][s]
                    self.Fg[i,j,self.nz-1][s] = self.fg[i,j,self.nz-2][s]

``convert_H_T()`` calculate the temperature 

.. code-block:: python

    @ti.func
    def convert_H_T(self,local_H):
        new_T=0.0
        #if local enthalpy is less than solid enthalpy 
        #T= enthalpy/specific heat
        if (local_H<self.H_s):
            new_T = local_H/self.Cp_s
        #if if local enthalpy is greater than liquid enthalpy 
        #T= Tliquid+(enthalpy-liquid enthalpy)/speific heat of liquid 
        elif (local_H>self.H_l):
            new_T = self.T_l+(local_H-self.H_l)/self.Cp_l
        #if if temperature is greater than solid temperature
        #T= Tsolid+(enthalpy-solid enthalpy)/(enthalpy of liquid-enthalpy of solid)*(temperature of liquid- temperature of solid)
        elif (self.T_l>self.T_s):
            new_T = self.T_s+(local_H-self.H_s)/(self.H_l-self.H_s)*(self.T_l-self.T_s)
        else:
        #else T= temperature of solid
            new_T = self.T_s

        return new_T

``convert_H_fl()`` calculate the volumn fraction of liquid

.. code-block:: python

    @ti.func
    def convert_H_fl(self,local_H):
        new_fl=0.0
        #if enthalpy is less than solid enthalpy 
        #it is zero
        if (local_H<self.H_s):
            new_fl = 0.0
        #if it is greater than liquid enthalpy
        #it is one
        elif (local_H>self.H_l):
            new_fl = 1.0
        #else 
        #it equals to (enthaply- soid enthaply)/(enthaply of liquid- enthalpy of solid)
        else:
            new_fl = (local_H-self.H_s)/(self.H_l-self.H_s)

        return new_fl

``convert_T_H()`` calculate the enthaply from temperature

.. code-block:: python

    @ti.func
    def convert_T_H(self,local_T):
        new_H = 0.0
        # calculate enthaply for three different conditions
        if (local_T<=self.T_s):
            new_H = self.Cp_s*local_T
        elif (local_T>self.T_l):
            new_H = (local_T-self.T_l)*self.Cp_l+self.H_l
        else:
            fluid_frc = (local_T-self.T_s)/(self.T_l-self.T_s)
            new_H = self.H_s*(1-fluid_frc) + self.H_l*fluid_frc
        return new_H

``convert_T_fl()`` calculate volumn fraction from temperature

.. code-block:: python

    @ti.func
    def convert_T_fl(self,local_T):
        new_fl = 0.0
        # calculate volumn fraction for three different conditions
        if (local_T<=self.T_s):
            new_fl = 0.0
        elif (local_T>=self.T_l):
            new_fl = 1.0
        elif (self.T_l>self.T_s):
            new_fl = (local_T-self.T_s)/(self.T_l-self.T_s)
        else:
            new_fl = 1.0

        return new_fl

``streaming3()`` calculate macroscopic variable 

.. code-block:: python

    @ti.kernel
    def streaming3(self):
        for i in ti.grouped(self.rho):
            self.forcexyz[i] = self.cal_local_force(i.x, i.y, i.z)
            #print(i.x, i.y, i.z)
            if ((self.solid[i]==0) or (self.rho_fl[i]>0.0)):
                self.rho[i] = 0
                self.v[i] = ti.Vector([0,0,0])
                self.f[i] = self.F[i]
                for s in ti.static(range(19)):
                    self.f[i][s] = self.f[i][s]*self.rho_fl[i]+self.w[s]*(1.0-self.rho_fl[i])
                #density for fluid
                self.rho[i] += self.f[i].sum()

                for s in ti.static(range(19)):
                    self.v[i] += self.e_f[s]*self.f[i][s]
                
                f = self.cal_local_force(i.x, i.y, i.z)
                #velocity for fluid 
                self.v[i] /= self.rho[i]
                self.v[i] += (f/2)/self.rho[i]
                
            else:
            #density and velocity for solid 
                self.rho[i] = 1.0
                self.v[i] = ti.Vector([0,0,0])

``streaming3()`` calculate enthalpy

.. code-block:: python

    @ti.kernel
    def streaming3_g(self):
        for i in ti.grouped(self.rho_T):
            self.rho_H[i] = 0.0
            #enthalpy here
            self.rho_H[i] = self.Fg[i].sum()
            #for s in ti.static(range(19)):
            #    self.rho_H[i] += self.Fg[i][s]
            self.fg[i] = self.Fg[i]

``update_T_fl()`` calculate volumn fraction and temperature

.. code-block:: python

    @ti.kernel
    def update_T_fl(self):
        for I in ti.grouped(self.rho_T):
            self.rho_T[I] = self.convert_H_T(self.rho_H[I])
            self.rho_fl[I] = self.convert_H_fl(self.rho_H[I])
            if (self.solid[I]>0):
                self.rho_fl[I] = 0.0

``init_solute_simulation()`` initialize the solute simulation

.. code-block:: python

    def init_solute_simulation(self):
    
        self.init_simulation()
        self.update_H_sl()
        #ethalpy
        self.init_H()
        #volumn fraction
        self.init_fl()
        #thermal distribution function
        self.init_fg()
    
``init_concentration(filename)`` import concentration data from file

.. code-block:: python

    def init_concentration(self,filename):
        in_dat = np.loadtxt(filename)
        in_dat = np.reshape(in_dat, (self.nx,self.ny,self.nz),order='F')
        self.rho_T.from_numpy(in_dat)

this 

.. code-block:: python

    def step(self):
        self.colission()
        self.colission_g()
        
        self.streaming1()
        self.streaming1_g()

        self.Boundary_condition()
        self.BC_concentration()

        self.streaming3_g()
        self.streaming3()
        self.streaming3_g()

        self.update_T_fl()

this

.. code-block:: python

    def export_VTK(self, n):
        gridToVTK(
                "./LB_SingelPhase_"+str(n),
                self.x,
                self.y,
                self.z,
                #cellData={"pressure": pressure},
                pointData={ "Solid": np.ascontiguousarray(self.solid.to_numpy()),
                            "rho": np.ascontiguousarray(self.rho.to_numpy()),
                            "Solid_Liquid": np.ascontiguousarray(self.rho_fl.to_numpy()),
                            "Tempreture": np.ascontiguousarray(self.rho_T.to_numpy()),
                            "Entropy": np.ascontiguousarray(self.rho_H.to_numpy()),
                            "velocity": (   np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,0]), 
                                            np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,1]),
                                            np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:self.nz,2])),
                            "Force": (      np.ascontiguousarray(self.forcexyz.to_numpy()[0:self.nx,0:self.ny,0:self.nz,0]), 
                                            np.ascontiguousarray(self.forcexyz.to_numpy()[0:self.nx,0:self.ny,0:self.nz,1]),
                                            np.ascontiguousarray(self.forcexyz.to_numpy()[0:self.nx,0:self.ny,0:self.nz,2]))
                            }
            )   

this