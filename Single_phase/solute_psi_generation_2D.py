import numpy as np
import math


# INPUT STL FILE NAME
output_name = 'geo_cavity.dat'
output_name2 = 'psi.dat'

dnx, dny, dnz = 50, 50, 5

#==========================================================
#           Geometry
#==========================================================
out_dat = np.zeros((dnx,dny,dnz))

#=======Can define some geometry here to out_dat=========
out_dat[:,0,:] = 1
out_dat[:,-1,:] = 1
#out_dat[:,0,:] = 1
#out_dat[:,-1,:] = 1
#out_dat[:,:,0] = 1
#out_dat[:,:,-1] = 1

out_dat = out_dat.reshape(out_dat.size, order = 'F')
np.savetxt(output_name,out_dat.T,fmt='%d')


#=========================================================
#           temprerature field
#=========================================================

out_dat = np.zeros((dnx,dny,dnz))+20

#=======Can define some geometry here to out_dat=========
out_dat[10:20,:,:] = 35
#out_dat[:,-1,:] = 10


#=========================================================

out_dat = out_dat.reshape(out_dat.size, order = 'F')
np.savetxt(output_name2,out_dat.T,fmt='%f')

