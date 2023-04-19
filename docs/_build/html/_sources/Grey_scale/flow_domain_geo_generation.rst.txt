flow_domain_geo_generation
=================================

This file output geometry data

.. code-block:: python

    #import numpy and math packahe
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
    #create np matrix with dnx*dny*dnz zero
    out_dat = np.zeros((dnx,dny,dnz))

    #=======Can define some geometry here to out_dat=========
    #out_dat[1,:,:] = 1

    #=========================================================
    #reshape out_dat with column major
    out_dat = out_dat.reshape(out_dat.size, order = 'F')
    #save the file with the transfer of out_dat based on integer type
    np.savetxt(output_name,out_dat.T,fmt='%d')