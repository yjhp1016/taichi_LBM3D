import numpy as np
import math


# INPUT STL FILE NAME
output_name = 'geo.dat'

# POINT SEARCHING RESOLUTION IN X direction, Y,Z direction will be calculate by the code
# the bigger value ~ more points will be found inside STL
dnx, dny, dnz = 60, 60, 60


#==========================================================
#           DO NOT CHANGE BELOW
#==========================================================

out_dat = np.zeros((dnx,dny,dnz))

#=======Can define some geometry here to out_dat=========
#out_dat[1,:,:] = 1

#=========================================================

out_dat = out_dat.reshape(out_dat.size, order = 'F')


np.savetxt(output_name,out_dat.T,fmt='%d')
