flow_domain_geo_generation_2D
=================================

This file generate geometry file for solver to read

.. code-block:: python 

    #import certain module
    import numpy as np
    import math


    #define the input file name
    # INPUT STL FILE NAME
    output_name = 'geo_cavity.dat'

    #define the grid resolution
    dnx, dny, dnz = 50, 50, 50

    #==========================================================
    #           DO NOT CHANGE BELOW
    #==========================================================

    #define an matrix dnx*dny*dnz with zero values
    out_dat = np.zeros((dnx,dny,dnz))

    #=======Can define some geometry here to out_dat=========
    #define the boundary to be solid 
    out_dat[0,:,:] = 1
    #cout_dat[:,:,0] = 1
    out_dat[:,0,:] = 1
    out_dat[:,-1,:] = 1
    out_dat[:,:,0] = 1
    out_dat[:,:,-1] = 1

    #=========================================================
    #reshape the data to be column major
    out_dat = out_dat.reshape(out_dat.size, order = 'F')


    #output the transfer of out_dat to the file with integer type
    np.savetxt(output_name,out_dat.T,fmt='%d')