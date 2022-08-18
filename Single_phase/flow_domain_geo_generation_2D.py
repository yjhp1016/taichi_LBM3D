import numpy as np
import math


# INPUT STL FILE NAME
output_name = 'geo_cavity.dat'


dnx, dny, dnz = 50, 50, 50

#==========================================================
#           DO NOT CHANGE BELOW
#==========================================================

out_dat = np.zeros((dnx,dny,dnz))

#=======Can define some geometry here to out_dat=========
out_dat[0,:,:] = 1
#cout_dat[:,:,0] = 1
out_dat[:,0,:] = 1
out_dat[:,-1,:] = 1
out_dat[:,:,0] = 1
out_dat[:,:,-1] = 1

#=========================================================

out_dat = out_dat.reshape(out_dat.size, order = 'F')


np.savetxt(output_name,out_dat.T,fmt='%d')
