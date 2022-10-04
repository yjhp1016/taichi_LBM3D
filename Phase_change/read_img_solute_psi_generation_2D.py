import numpy as np

import copy

img_in = np.loadtxt('C:\\Users\\J0424810\\Downloads\\Work\\YF\\melt\\test1\\pic1.txt')

# INPUT STL FILE NAME
output_name = 'geo_cavity.dat'
output_name2 = 'psi.dat'

dnx, dny, dnz = img_in.shape[0], img_in.shape[1], 3

#==========================================================
#           Geometry
#==========================================================
out_dat = np.zeros((dnx,dny,dnz))

#=======Can define some geometry here to out_dat=========
for i in range(dnz):
    out_dat[:,:,i] = 255-img_in
#out_dat[:,0,:] = 1
#out_dat[:,-1,:] = 1
#out_dat[:,0,:] = 1
#out_dat[:,-1,:] = 1
#out_dat[:,:,0] = 1
#out_dat[:,:,-1] = 1

out_dat2 = copy.deepcopy(out_dat)

I1 = out_dat>180
I2 = out_dat<=180
out_dat[I1] = 1
out_dat[~I1] = 0

out_dat = out_dat.reshape(out_dat.size, order = 'F')
np.savetxt(output_name,out_dat.T,fmt='%d')


#=========================================================
#           temprerature field
#=========================================================

#out_dat2 = np.zeros((dnx,dny,dnz))#+np.random.rand(dnx, dny,dnz)+20

#=======Can define some geometry here to out_dat=========
I1 = out_dat2>30
out_dat2[I1] = 10.0
out_dat2[~I1] = 20.0


#=========================================================

out_dat2 = out_dat2.reshape(out_dat2.size, order = 'F')
np.savetxt(output_name2,out_dat2.T,fmt='%f')

print(dnx, dny,dnz)

