from vtk_writer import write_basic_mesh

import numpy
Verts = numpy.array([[0.0,0.0],
                     [1.0,0.0],
                     [2.0,0.0],
                     [0.0,1.0],
                     [1.0,1.0],
                     [2.0,1.0],
                     [0.0,2.0],
                     [1.0,2.0],
                     [2.0,2.0],
                     [0.0,3.0],
                     [1.0,3.0],
                     [2.0,3.0]])
E2V = numpy.array([[0,4,3],
                   [0,1,4],
                   [1,5,4],
                   [1,2,5],
                   [3,7,6],
                   [3,4,7],
                   [4,8,7],
                   [4,5,8],
                   [6,10,9],
                   [6,7,10],
                   [7,11,10],
                   [7,8,11]])

pdata=numpy.ones((12,2))
pvdata=numpy.ones((12*3,2))
cdata=numpy.ones((12,2))
cvdata=numpy.ones((3*12,2))

write_basic_mesh(Verts, E2V=E2V, mesh_type='tri',\
        pdata=pdata, pvdata=pvdata, cdata=cdata, cvdata=cvdata,\
        fname='test.vtu')

z = numpy.zeros((Verts.shape[0],))

# plot x, y scalar values
pdata=Verts                                          
# rotation vector [x,-y,0]
pvdata=numpy.vstack((Verts[:,0],-1.0*Verts[:,1],z)).T.ravel()
# average index for a cell
cdata=E2V.mean(axis=1)                               

write_basic_mesh(Verts, E2V=E2V, mesh_type='tri',\
        pdata=pdata, pvdata=pvdata, cdata=cdata,\
        fname='test2.vtu')
