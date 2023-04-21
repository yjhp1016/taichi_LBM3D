lbm_solver_3d_sparse
=================================


This solver is almost similar to lbm_solver_3d expect the sparse definition of some varible:

.. code-block:: python

    f = ti.field(ti.f32)
    F = ti.field(ti.f32)
    rho = ti.field(ti.f32)
    v = ti.Vector.field(3, ti.f32)
    n_mem_partition = 3

    cell1 = ti.root.pointer(ti.ijk, (nx//n_mem_partition+1,ny//n_mem_partition+1,nz//n_mem_partition+1))
    cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(rho)
    cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(v)

    cell2 = ti.root.pointer(ti.ijkl,(nx//3+1,ny//3+1,nz//3+1,1))
    cell2.dense(ti.ijkl,(n_mem_partition,n_mem_partition,n_mem_partition,19)).place(f)
    cell2.dense(ti.ijkl,(n_mem_partition,n_mem_partition,n_mem_partition,19)).place(F)

It use a pointer and certain block to divide the region and then place different varible on the block which make the storage
sparse.