lbm_solver_3d_2phase_sparse
========================================

This file is almost the same as the ``lbm_solver_3d_2phase.py`` file execpt sparse storage definition of some varibles

.. code-block:: python

    # Sparse Storage memory allocation
    f = ti.field(ti.f32)
    F = ti.field(ti.f32)
    rho = ti.field(ti.f32)
    v = ti.Vector.field(3, ti.f32)
    rhor = ti.field(ti.f32)
    rhob = ti.field(ti.f32)
    rho_r = ti.field(ti.f32)
    rho_b = ti.field(ti.f32)
    n_mem_partition = 3

    cell1 = ti.root.pointer(ti.ijk, (nx//n_mem_partition+1,ny//n_mem_partition+1,nz//n_mem_partition+1))
    cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(rho)
    cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(v)
    cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(rhor)
    cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(rhob)
    cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(rho_r)
    cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(rho_b)


    cell2 = ti.root.pointer(ti.ijkl,(nx//3+1,ny//3+1,nz//3+1,1))
    cell2.dense(ti.ijkl,(n_mem_partition,n_mem_partition,n_mem_partition,19)).place(f)
    cell2.dense(ti.ijkl,(n_mem_partition,n_mem_partition,n_mem_partition,19)).place(F)

Above code snippts define the sparse storage of some varibles