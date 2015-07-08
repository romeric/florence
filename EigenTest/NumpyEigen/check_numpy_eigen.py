import numpy_eigen as ne
import numpy as np
from time import time 

from ReadSalomeMesh import ReadMesh

# mesh=ReadMesh("Mesh_Annular_Circle_75.dat","tri")
# mesh=ReadMesh("/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/Mesh_Annular_Circle_75.dat","tri")
# mesh=ReadMesh("/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/Mesh_Annular_Circle_382526.dat","tri")
mesh=ReadMesh("/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/Mesh_Annular_Circle_6135253.dat","tri")
print 'mesh read!'

t1=time()
dirichletbc = ne.copy_to_std_vector(mesh.elements,mesh.points)
print time()-t1 
# print dirichletbc
