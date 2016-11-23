"""VTK output functions.

Use the XML VTK format for unstructured meshes (.vtu)

See here for a guide:  http://www.vtk.org/pdf/file-formats.pdf

Luke Olson 20090309
http://www.cs.uiuc.edu/homes/lukeo
"""

__docformat__ = "restructuredtext en"

__all__ = ['write_vtu','write_basic_mesh']

import xml.dom.minidom
import numpy

def write_vtu(Verts, Cells, pdata=None, pvdata=None, cdata=None, cvdata=None, fname='output.vtu'):
    """
    Write a .vtu file in xml format

    Parameters
    ----------
    fname : {string}
        file to be written, e.g. 'mymesh.vtu'
    Verts : {array}
        Ndof x 3 (if 2, then expanded by 0)
        list of (x,y,z) point coordinates
    Cells : {dictionary}
        Dictionary of with the keys
    pdata : {array}
        Ndof x Nfields array of scalar values for the vertices
    pvdata : {array}
        Nfields*3 x Ndof array of vector values for the vertices
    cdata : {dictionary}
        scalar valued cell data
    cvdata : {dictionary}
        vector valued cell data

    Returns
    -------
     writes a .vtu file for use in Paraview

    Notes
    -----
    - Poly data not supported 
    - Non-Poly data is stored in Numpy array: Ncell x vtk_cell_info
    - Each I1 must be >=3
    - pdata = Ndof x Nfields
    - pvdata = 3*Ndof x Nfields
    - cdata,cvdata = list of dictionaries in the form of Cells


    =====  ========================= ============= ===
    keys   type                      n points      dim
    =====  ========================= ============= ===
       1   VTK_VERTEX:               1 point        2d
       2   VTK_POLY_VERTEX:          n points       2d
       3   VTK_LINE:                 2 points       2d
       4   VTK_POLY_LINE:            n+1 points     2d
       5   VTK_TRIANGLE:             3 points       2d
       6   VTK_TRIANGLE_STRIP:       n+2 points     2d
       7   VTK_POLYGON:              n points       2d
       8   VTK_PIXEL:                4 points       2d
       9   VTK_QUAD:                 4 points       2d
       10  VTK_TETRA:                4 points       3d
       11  VTK_VOXEL:                8 points       3d
       12  VTK_HEXAHEDRON:           8 points       3d
       13  VTK_WEDGE:                6 points       3d
       14  VTK_PYRAMID:              5 points       3d
       24  VTK_QUADRATIC_TETRA       10 points      3d
    =====  ========================= ============= ===

    Examples
    --------
    >>> import numpy
    >>> Verts = numpy.array([[0.0,0.0],
    ...                      [1.0,0.0],
    ...                      [2.0,0.0],
    ...                      [0.0,1.0],
    ...                      [1.0,1.0],
    ...                      [2.0,1.0],
    ...                      [0.0,2.0],
    ...                      [1.0,2.0],
    ...                      [2.0,2.0],
    ...                      [0.0,3.0],
    ...                      [1.0,3.0],
    ...                      [2.0,3.0]])
    >>> E2V = numpy.array([[0,4,3],
    ...                    [0,1,4],
    ...                    [1,5,4],
    ...                    [1,2,5],
    ...                    [3,7,6],
    ...                    [3,4,7],
    ...                    [4,8,7],
    ...                    [4,5,8],
    ...                    [6,10,9],
    ...                    [6,7,10],
    ...                    [7,11,10],
    ...                    [7,8,11]])
    >>> E2edge = numpy.array([[0,1]])
    >>> E2point = numpy.array([2,3,4,5])
    >>> Cells = {5:E2V,3:E2edge,1:E2point}
    >>> pdata=numpy.ones((12,2))
    >>> pvdata=numpy.ones((12*3,2))
    >>> cdata={5:numpy.ones((12,2)),3:numpy.ones((1,2)),1:numpy.ones((4,2))}
    >>> cvdata={5:numpy.ones((3*12,2)),3:numpy.ones((3*1,2)),1:numpy.ones((3*4,2))}
    >>> write_vtu(Verts=Verts, Cells=Cells, fname='test.vtu')

    See Also
    --------
    write_basic_mesh
       
    """
    # number of indices per cell for each cell type
    # vtk_cell_info = [-1, 1, None, 2, None, 3, None, None, 4, 4, 4, 8, 8, 6, 5]
    vtk_cell_info = [-1, 1, None, 2, None, 3, None, None, 4, 4, 4, 8, 8, 6, 5,
                        None, None, None, None, None, None, 3, 6, 8, 10, 20]

    # check fname
    if type(fname) is str:
        try:
            fname = open(fname,'w')
        except IOError, (errno, strerror):
            print ".vtu error (%s): %s" % (errno, strerror)
    else:
        raise ValueError('fname is assumed to be a string')

    # check Verts
    # get dimension and verify that it's 3d data
    Ndof,dim = Verts.shape
    if dim==2:
        # always use 3d coordinates (x,y) -> (x,y,0)
        Verts = numpy.hstack((Verts,numpy.zeros((Ndof,1))))

    # check Cells
    # keys must ve valid (integer and not "None" in vtk_cell_info)
    # Cell data can't be empty for a non empty key
    for key in Cells:
        if ((type(key) != int) or (key not in range(1,26))):
            raise ValueError('cell array must have positive integer keys in [1,25]')
        if (vtk_cell_info[key] == None) and (Cells[key] != None):
            # Poly data
            raise NotImplementedError('Poly Data not implemented yet')
        if Cells[key] is None:
            raise ValueError('cell array cannot be empty for key %d'%(key))
        if numpy.ndim(Cells[key])!=2:
            Cells[key] = Cells[key].reshape((Cells[key].size,1))
        if vtk_cell_info[key] != Cells[key].shape[1]:
            raise ValueError('cell array has %d columns, expected %d' % (offset, vtk_cell_info[key]) )

    # check pdata
    # must be Ndof x n_pdata
    n_pdata = 0
    if pdata is not None:
        if numpy.ndim(pdata)>1:
            n_pdata=pdata.shape[1]
        else:
            n_pdata = 1
            pdata = pdata.reshape((pdata.size,1)) 
        if pdata.shape[0] != Ndof:
            raise ValueError('pdata array should be of length %d (it is now %d)'%(Ndof,pdata.shape[0]))

    # check pvdata
    # must be 3*Ndof x n_pvdata
    n_pvdata = 0
    if pvdata != None:
        if numpy.ndim(pvdata)>1:
            n_pvdata = pvdata.shape[1]
        else:
            n_pvdata = 1
            pvdata = pvdata.reshape((pvdata.size,1))
        if pvdata.shape[0] != 3*Ndof:
            raise ValueError('pvdata array should be of size %d (or multiples) (it is now %d)'%(Ndof*3,pvdata.shape[0]))

    # check cdata
    # must be NCells x n_cdata for each key
    n_cdata = 0
    if cdata !=None:
        for key in Cells:   # all valid now
            if numpy.ndim(cdata[key])>1:
                if n_cdata==0:
                    n_cdata=cdata[key].shape[1]
                elif n_cdata!=cdata[key].shape[1]:
                    raise ValueError('cdata dimension problem')
            else:
                n_cdata=1
                cdata[key] = cdata[key].reshape((cdata[key].size,1))
            if cdata[key].shape[0]!=Cells[key].shape[0]:
                raise ValueError('size mismatch with cdata %d and Cells %d'%(cdata[key].shape[0],Cells[key].shape[0]))
            if cdata[key] == None:
                raise ValueError('cdata array cannot be empty for key %d'%(key))

    # check cvdata
    # must be NCells*3 x n_cdata for each key
    n_cvdata = 0
    if cvdata !=None:
        for key in Cells:   # all valid now
            if numpy.ndim(cvdata[key])>1:
                if n_cvdata==0:
                    n_cvdata=cvdata[key].shape[1]
                elif n_cvdata!=cvdata[key].shape[1]:
                    raise ValueError('cvdata dimension problem')
            else:
                n_cvdata=1
                cvdata[key] = cvdata[key].reshape((cvdata[key].size,1))
            if cvdata[key].shape[0]!=3*Cells[key].shape[0]:
                raise ValueError('size mismatch with cvdata and Cells')
            if cvdata[key] == None:
                raise ValueError('cvdata array cannot be empty for key %d'%(key))
            
    Ncells = 0
    idx_min = 1

    cell_ind    = []
    cell_offset = [] #= numpy.zeros((Ncells,1),dtype=uint8) # offsets are zero indexed
    cell_type   = [] #= numpy.zeros((Ncells,1),dtype=uint8)

    cdata_all = None
    cvdata_all = None
    for key in Cells:
            # non-Poly data
            sz = Cells[key].shape[0]
            offset = Cells[key].shape[1]

            Ncells += sz
            cell_ind    = numpy.hstack((cell_ind,Cells[key].ravel()))
            cell_offset = numpy.hstack((cell_offset,offset*numpy.ones((sz,),dtype='uint8')))
            cell_type   = numpy.hstack((cell_type,key*numpy.ones((sz,),dtype='uint8')))
            
            if cdata != None:
                if cdata_all==None:
                    cdata_all=cdata[key]
                else:
                    cdata_all = numpy.vstack((cdata_all,cdata[key]))

            if cvdata != None:
                if cvdata_all==None:
                    cvdata_all=cvdata[key]
                else:
                    cvdata_all = numpy.vstack((cvdata_all,cvdata[key]))


    # doc element
    doc = xml.dom.minidom.Document()

    # vtk element
    root = doc.createElementNS('VTK', 'VTKFile')
    d = {'type':'UnstructuredGrid', 'version':'0.1', 'byte_order':'LittleEndian'}
    set_attributes(d,root)

    # unstructured element
    grid = doc.createElementNS('VTK', 'UnstructuredGrid')

    # piece element
    piece = doc.createElementNS('VTK', 'Piece')
    d = {'NumberOfPoints':str(Ndof),'NumberOfCells':str(Ncells)}
    set_attributes(d,piece)

    ## POINTS
    # points element
    points = doc.createElementNS('VTK', 'Points')
    # data element
    points_data = doc.createElementNS('VTK', 'DataArray')
    d = {'type':'Float32', 'Name':'vertices', 'NumberOfComponents':'3', 'format':'ascii'}
    set_attributes(d,points_data)
    # string for data element
    points_data_str = doc.createTextNode(a2s(Verts))

    ## CELLS
    # points element
    cells = doc.createElementNS('VTK', 'Cells')
    # data element
    cells_data = doc.createElementNS('VTK', 'DataArray')
    d = {'type':'Int32', 'Name':'connectivity', 'format':'ascii'}
    # d = {'type':'UInt8', 'Name':'connectivity', 'format':'ascii'}
    set_attributes(d,cells_data)
    # string for data element
    cells_data_str = doc.createTextNode(a2s(cell_ind))
    # offset data element
    cells_offset_data = doc.createElementNS('VTK', 'DataArray')
    d = {'type':'Int32', 'Name':'offsets', 'format':'ascii'}
    # d = {'type':'UInt8', 'Name':'offsets', 'format':'ascii'}
    set_attributes(d,cells_offset_data)
    # string for data element
    cells_offset_data_str = doc.createTextNode(a2s(cell_offset.cumsum()))
    # offset data element
    cells_type_data = doc.createElementNS('VTK', 'DataArray')
    d = {'type':'UInt8', 'Name':'types', 'format':'ascii'}
    set_attributes(d,cells_type_data)
    # string for data element
    cells_type_data_str = doc.createTextNode(a2s(cell_type))

    ## POINT DATA
    pointdata = doc.createElementNS('VTK', 'PointData')
    # pdata
    pdata_obj=[]
    pdata_str=[]
    for i in range(0,n_pdata):
        pdata_obj.append(doc.createElementNS('VTK', 'DataArray'))
        # d = {'type':'Float32', 'Name':'pdata %d'%(i), 'NumberOfComponents':'1', 'format':'ascii'}
        d = {'type':'Float32', 'Name':'Uz %d'%(i), 'NumberOfComponents':'1', 'format':'ascii'}
        set_attributes(d,pdata_obj[i])
        pdata_str.append(doc.createTextNode(a2s(pdata[:,i])))
    # pvdata
    pvdata_obj=[]
    pvdata_str=[]
    for i in range(0,n_pvdata):
        pvdata_obj.append(doc.createElementNS('VTK', 'DataArray'))
        d = {'type':'Float32', 'Name':'pvdata %d'%(i), 'NumberOfComponents':'3', 'format':'ascii'}
        set_attributes(d,pvdata_obj[i])
        pvdata_str.append(doc.createTextNode(a2s(pvdata[:,i])))

    ## CELL DATA
    celldata = doc.createElementNS('VTK', 'CellData')
    # cdata
    cdata_obj=[]
    cdata_str=[]
    for i in range(0,n_cdata):
        cdata_obj.append(doc.createElementNS('VTK', 'DataArray'))
        d = {'type':'Float32', 'Name':'cdata %d'%(i), 'NumberOfComponents':'1', 'format':'ascii'}
        set_attributes(d,cdata_obj[i])
        cdata_str.append(doc.createTextNode(a2s(cdata_all[:,i])))
    # cvdata
    cvdata_obj=[]
    cvdata_str=[]
    for i in range(0,n_cvdata):
        cvdata_obj.append(doc.createElementNS('VTK', 'DataArray'))
        d = {'type':'Float32', 'Name':'cvdata %d'%(i), 'NumberOfComponents':'3', 'format':'ascii'}
        set_attributes(d,cvdata_obj[i])
        cvdata_str.append(doc.createTextNode(a2s(cvdata_all[:,i])))

    doc.appendChild(root)
    root.appendChild(grid)
    grid.appendChild(piece)

    piece.appendChild(points)
    points.appendChild(points_data)
    points_data.appendChild(points_data_str)

    piece.appendChild(cells)
    cells.appendChild(cells_data)
    cells.appendChild(cells_offset_data)
    cells.appendChild(cells_type_data)
    cells_data.appendChild(cells_data_str)
    cells_offset_data.appendChild(cells_offset_data_str)
    cells_type_data.appendChild(cells_type_data_str)

    piece.appendChild(pointdata)
    for i in range(0,n_pdata):
        pointdata.appendChild(pdata_obj[i])
        pdata_obj[i].appendChild(pdata_str[i])
    for i in range(0,n_pvdata):
        pointdata.appendChild(pvdata_obj[i])
        pvdata_obj[i].appendChild(pvdata_str[i])

    piece.appendChild(celldata)
    for i in range(0,n_cdata):
        celldata.appendChild(cdata_obj[i])
        cdata_obj[i].appendChild(cdata_str[i])
    for i in range(0,n_cvdata):
        celldata.appendChild(cvdata_obj[i])
        cvdata_obj[i].appendChild(cvdata_str[i])

    doc.writexml(fname, newl='\n')
    fname.close()


def write_basic_mesh(Verts, E2V=None, mesh_type='tri', \
        pdata=None, pvdata=None, \
        cdata=None, cvdata=None, fname='output.vtu'):
    """
    Write mesh file for basic types of elements

    Parameters
    ----------
    fname : {string}
        file to be written, e.g. 'mymesh.vtu'
    Verts : {array}
        coordinate array (N x D)
    E2V : {array}
        element index array (Nel x Nelnodes)
    mesh_type : {string}
        type of elements: tri, quad, tet, hex (all 3d)
    pdata : {array}
        scalar data on vertices (N x Nfields)
    pvdata : {array}
        vector data on vertices (3*Nfields x N)
    cdata : {array}
        scalar data on cells (Nfields x Nel)
    cvdata : {array}
        vector data on cells (3*Nfields x Nel)

    Returns
    -------
    writes a .vtu file for use in Paraview

    Notes
    -----
    The difference between write_basic_mesh and write_vtu is that write_vtu is
    more general and requires dictionaries of cell information.
    write_basic_mesh calls write_vtu

    Examples
    --------
    >>> import numpy
    >>> Verts = numpy.array([[0.0,0.0],
    ...                      [1.0,0.0],
    ...                      [2.0,0.0],
    ...                      [0.0,1.0],
    ...                      [1.0,1.0],
    ...                      [2.0,1.0],
    ...                      [0.0,2.0],
    ...                      [1.0,2.0],
    ...                      [2.0,2.0],
    ...                      [0.0,3.0],
    ...                      [1.0,3.0],
    ...                      [2.0,3.0]])
    >>> E2V = numpy.array([[0,4,3],
    ...                    [0,1,4],
    ...                    [1,5,4],
    ...                    [1,2,5],
    ...                    [3,7,6],
    ...                    [3,4,7],
    ...                    [4,8,7],
    ...                    [4,5,8],
    ...                    [6,10,9],
    ...                    [6,7,10],
    ...                    [7,11,10],
    ...                    [7,8,11]])
    >>> pdata=numpy.ones((12,2))
    >>> pvdata=numpy.ones((12*3,2))
    >>> cdata=numpy.ones((12,2))
    >>> cvdata=numpy.ones((3*12,2))
    >>> write_basic_mesh(Verts, E2V=E2V, mesh_type='tri',pdata=pdata, pvdata=pvdata, cdata=cdata, cvdata=cvdata, fname='test.vtu')

    See Also
    --------
    write_vtu

    """
    if E2V is None:
        mesh_type='vertex'

    map_type_to_key = {'vertex':1, 'tri':5, 'quad':9, 'tet':10, 'hex':12}

    if mesh_type not in map_type_to_key:
        raise ValueError('unknown mesh_type=%s' % mesh_type)
    
    key = map_type_to_key[mesh_type]

    if mesh_type=='vertex':
        E2V = { key : numpy.arange(0,Verts.shape[0]).reshape((Verts.shape[0],1))}
    else:
        E2V = { key : E2V }

    if cdata != None:
        cdata = {key: cdata} 
    
    if cvdata != None:
        cvdata = {key: cvdata}

    write_vtu(Verts=Verts, Cells=E2V, pdata=pdata, pvdata=pvdata, \
            cdata=cdata, cvdata=cvdata, fname=fname)


# ---------------------------------------
def set_attributes(d,elm):
    """
    helper function: Set attributes from dictionary of values
    """
    for key in d:
        elm.setAttribute(key,d[key])

def a2s(a):
    """
    helper function: Convert to string
    """
    str=''
    return str.join(['%g '%(v) for v in a.ravel()])
