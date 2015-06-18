# gmsh reader
# neu writer
#
# * handles triangles (2d), tets(3d)
import numpy
import scipy
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, triu, eye
import sys


class Mesh:
    """
    Store the verts and elements and physical data

    attributes
    ----------
    Verts : array
        array of 3d coordinates (npts x 3)
    Elmts : dict
        dictionary of tuples
        (rank 1 array of physical ids, rank 2 array of element to vertex ids
        (Nel x ppe)) each array in the tuple is of length nElmts Phys : dict
        keys and names

    methods
    -------
    read_msh:
        read a 2.0 ascii gmsh file
    write_neu:
        write a gambit neutral file. works for tets, tris in 3d and 2d
    write_vtu:
        write VTK file (calling vtk_writer.py)
    """
    def __init__(self):

        self.Verts = []
        self.Elmts = {}
        self.Phys = {}

        self.npts = 0
        self.nElmts = {}
        self.nprops = 0

        self._elm_types()   # sets elm_type

        self.meshname = ""

    def read_msh(self, mshfile):
        """Read a Gmsh .msh file.

        Reads Gmsh 2.0 mesh files
        """
        self.meshname = mshfile
        try:
            fid = open(mshfile, "r")
        except IOError:
            print "File '%s' not found." % (filename)
            sys.exit()

        line = 'start'
        while line:
            line = fid.readline()

            if line.find('$MeshFormat') == 0:
                line = fid.readline()
                if line.split()[0][0] is not '2':
                    print "wrong gmsh version"
                    sys.exit()
                line = fid.readline()
                if line.find('$EndMeshFormat') != 0:
                    raise ValueError('expecting EndMeshFormat')

            if line.find('$PhysicalNames') == 0:
                line = fid.readline()
                self.nprops = int(line.split()[0])
                for i in range(0, self.nprops):
                    line = fid.readline()
                    newkey = int(line.split()[0])
                    qstart = line.find('"')+1
                    qend = line.find('"', -1, 0)-1
                    self.Phys[newkey] = line[qstart:qend]
                line = fid.readline()
                if line.find('$EndPhysicalNames') != 0:
                    raise ValueError('expecting EndPhysicalNames')

            if line.find('$Nodes') == 0:
                line = fid.readline()
                self.npts = int(line.split()[0])
                self.Verts = numpy.zeros((self.npts, 3), dtype=float)
                for i in range(0, self.npts):
                    line = fid.readline()
                    data = line.split()
                    idx = int(data[0])-1  # fix gmsh 1-based indexing
                    if i != idx:
                        raise ValueError('problem with vertex ids')
                    self.Verts[idx, :] = map(float, data[1:])
                line = fid.readline()
                if line.find('$EndNodes') != 0:
                    raise ValueError('expecting EndNodes')

            if line.find('$Elements') == 0:
                line = fid.readline()
                self.nel = int(line.split()[0])
                for i in range(0, self.nel):
                    line = fid.readline()
                    data = line.split()
                    idx = int(data[0])-1  # fix gmsh 1-based indexing
                    if i != idx:
                        raise ValueError('problem with elements ids')
                    etype = int(data[1])           # element type
                    nnodes = self.elm_type[etype]   # lookup number of nodes
                    ntags = int(data[2])           # number of tags following
                    k = 3
                    if ntags > 0:                   # set physical id
                        physid = int(data[k])
                        if physid not in self.Phys:
                            self.Phys[physid] = 'Physical Entity %d' % physid
                            self.nprops += 1
                        k += ntags

                    verts = map(int, data[k:])
                    verts = numpy.array(verts)-1  # fixe gmsh 1-based index

                    if (etype not in self.Elmts) or\
                            (len(self.Elmts[etype]) == 0):
                        # initialize
                        self.Elmts[etype] = (physid, verts)
                        self.nElmts[etype] = 1
                    else:
                        # append
                        self.Elmts[etype] = \
                            (numpy.hstack((self.Elmts[etype][0], physid)),
                             numpy.vstack((self.Elmts[etype][1], verts)))
                        self.nElmts[etype] += 1

                line = fid.readline()
                if line.find('$EndElements') != 0:
                    raise ValueError('expecting EndElements')
        fid.close()

    def _find_EF(self, vlist, E):
        for i in range(0, E.shape[0]):
            enodes = E[i, :]
            if len(numpy.intersect1d_nu(vlist, enodes)) == len(vlist):
                # found element.  now the face
                missing_node = numpy.setdiff1d(enodes, vlist)
                loc = numpy.where(enodes == missing_node)[0][0]

                # determine face from missing id
                if len(enodes) == 3:  # tri
                    face_map = {0: 1, 1: 2, 2: 0}
                if len(enodes) == 4:  # tet
                    face_map = {0: 2, 1: 3, 2: 1, 3: 0}

                return i, face_map[loc]

    def write_vtu(self, fname=None):
        if fname is None:
            fname = self.meshname.split('.')[0] + '.vtu'

        from vtk_writer import write_vtu
        vtk_id = {1: 3, 2: 5, 4: 10, 15: 1}
        Cells = {}
        cdata = {}
        k = 0.0
        for g_id, E in self.Elmts.iteritems():
            k += 1.0
            if g_id not in vtk_id:
                raise NotImplementedError('vtk ids not yet implemented')
            Cells[vtk_id[g_id]] = E[1]
            cdata[vtk_id[g_id]] = k*numpy.ones((E[1].shape[0],))

        write_vtu(Verts=self.Verts, Cells=Cells, cdata=cdata, fname=fname)

    def write_neu(self, fname=None):
        """ works for tets, tris in 3d and 2d"""

        neu_id = {1: 3, 2: 3, 4: 6, 15: 1}
        neu_pts = {1: 2, 2: 3, 4: 4, 15: 1}

        if fname is None:
            fname = self.meshname.split('.')[0] + '.neu'

        if type(fname) is str:
            try:
                fid = open(fname, 'w')
            except IOError, (errno, strerror):
                print ".neu error (%s): %s" % (errno, strerror)
        else:
            raise ValueError('fname is assumed to be a string')

        if 4 not in self.Elmts:
            mesh_id = 4  # mesh elements are tets
            bc_id = 2   # bdy face elements are tris
            dim = 3
            print '... (neu file) assuming 3d, using tetrahedra'
        elif 2 not in self.Elmts:
            mesh_id = 2  # mesh elements are tris
            bc_id = 1   # bdy face elements are lines
            dim = 2
            print '... (neu file) assuming 2d, using triangles'
        else:
            raise ValueError('problem with finding elements for neu file')

        E = self.Elmts[mesh_id][1]
        nel = self.nElmts[mesh_id]
        if E.shape[0] != nel:
            raise ValueError('problem with element shape and nel')

        Eb = self.Elmts[bc_id][1]
        nelb = self.nElmts[bc_id]
        if Eb.shape[0] != nelb:
            raise ValueError('problem with element shape and nel')

        bd_id_list = self.Elmts[bc_id][0]
        bd_ids = numpy.unique(bd_id_list)
        nbc = len(bd_ids)

        # list of Elements on the bdy and corresponding face
        EF = numpy.zeros((nelb, 2), dtype=int)
        for i in range(0, nelb):
            vlist = Eb[i, :]
            el, face = self._find_EF(vlist, E)
            EF[i, :] = [el+1, face+1]
            #          ^^     ^^   neu is 1 based indexing

        fid.write('        CONTROL INFO 1.3.0\n')
        fid.write('** GAMBIT NEUTRAL FILE\n\n\n\n')
        fid.write('%10s%10s%10s%10s%10s%10s\n' %
                  ('NUMNP', 'NELEM', 'NGRPS', 'NBSETS', 'NDFCD', 'NDFVL'))
        data = (self.npts, nel, 0, nbc, dim, dim)
        fid.write('%10d%10d%10d%10d%10d%10d\n' % data)
        fid.write('ENDOFSECTION\n')
        fid.write('   NODAL COORDINATES 1.3.0\n')
        for i in range(0, self.npts):
            if dim == 2:
                fid.write('%d  %e  %e\n' %
                          (i+1, self.Verts[i, 0], self.Verts[i, 1]))
                                        # ^^^ neu is 1-based indexing
            else:
                fid.write('%d  %e  %e  %e\n' %
                          (i+1, self.Verts[i, 0], self.Verts[i, 1],
                           self.Verts[i, 2]))
                                        #     ^^^ neu is 1-based indexing
        fid.write('ENDOFSECTION\n')

        fid.write('ELEMENTS/CELLS 1.3.0\n')
        for i in range(0, nel):
            data = [i+1, neu_id[mesh_id], neu_pts[mesh_id]]
                #   ^^^ neu is 1-based indexing
            data.extend((E[i, :]+1).tolist())
                          #   ^^^ neu is 1-based indexing
            dstr = ''
            for d in data:
                dstr += ' %d' % d
            fid.write(dstr+'\n')
        fid.write('ENDOFSECTION\n')
        #    Write all the boundary condition blocks IMPORTANT, it is assumed
        #    that the the mesh_condition_name stores the BC type and the first
        #    number to follow the BC in the .neu file.  For circular
        #    boundaries, mesh_condition_name[i] = "circ  _radius-length_" For
        #    standard dirichlet boundaries, mesh_condition_name[i] = "diri 1".
        #    examples
        #    Wall 1
        #    Inflow 1
        #    Outflow 1
        #
        #    1 means cell
        #    0 means node
        for bcid in range(0, nbc):
            this_bdy = numpy.where(bd_id_list == bd_ids[bcid])[0]
            bnel = len(this_bdy)
            fid.write('   BOUNDARY CONDITIONS 1.3.0\n')
            fid.write('%10s   %d   %d   %d\n' %
                      (self.Phys[bd_ids[bcid]], bnel, 0, 0))
            for i in range(0, bnel):
                el = EF[this_bdy[i], 0]
                face = EF[this_bdy[i], 1]
                fid.write(' %d %d %d \n' % (el, neu_id[mesh_id], face))
            fid.write('ENDOFSECTION\n')

        fid.close()

    def _elm_types(self):
        elm_type = {}
        elm_type[1] = 2    # 2-node line
        elm_type[2] = 3    # 3-node triangle
        elm_type[3] = 4    # 4-node quadrangle
        elm_type[4] = 4    # 4-node tetrahedron
        elm_type[5] = 8    # 8-node hexahedron
        elm_type[6] = 6    # 6-node prism
        elm_type[7] = 5    # 5-node pyramid
        elm_type[8] = 3    # 3-node second order line
                            # (2 nodes at vertices and 1 with edge)
        elm_type[9] = 6    # 6-node second order triangle
                            # (3 nodes at vertices and 3 with edges)
        elm_type[10] = 9    # 9-node second order quadrangle
                            # (4 nodes at vertices,
                            #  4 with edges and 1 with face)
        elm_type[11] = 10   # 10-node second order tetrahedron
                            # (4 nodes at vertices and 6 with edges)
        elm_type[12] = 27   # 27-node second order hexahedron
                            # (8 nodes at vertices, 12 with edges,
                            #  6 with faces and 1 with volume)
        elm_type[13] = 18   # 18-node second order prism
                            # (6 nodes at vertices,
                            #  9 with edges and 3 with quadrangular faces)
        elm_type[14] = 14   # 14-node second order pyramid
                            # (5 nodes at vertices,
                            #  8 with edges and 1 with quadrangular face)
        elm_type[15] = 1    # 1-node point
        elm_type[16] = 8    # 8-node second order quadrangle
                            # (4 nodes at vertices and 4 with edges)
        elm_type[17] = 20   # 20-node second order hexahedron
                            # (8 nodes at vertices and 12 with edges)
        elm_type[18] = 15   # 15-node second order prism
                            # (6 nodes at vertices and 9 with edges)
        elm_type[19] = 13   # 13-node second order pyramid
                            # (5 nodes at vertices and 8 with edges)
        elm_type[20] = 9    # 9-node third order incomplete triangle
                            # (3 nodes at vertices, 6 with edges)
        elm_type[21] = 10   # 10-node third order triangle
                            # (3 nodes at vertices, 6 with edges, 1 with face)
        elm_type[22] = 12   # 12-node fourth order incomplete triangle
                            # (3 nodes at vertices, 9 with edges)
        elm_type[23] = 15   # 15-node fourth order triangle
                            # (3 nodes at vertices, 9 with edges, 3 with face)
        elm_type[24] = 15   # 15-node fifth order incomplete triangle
                            # (3 nodes at vertices, 12 with edges)
        elm_type[25] = 21   # 21-node fifth order complete triangle
                            # (3 nodes at vertices, 12 with edges, 6 with face)
        elm_type[26] = 4    # 4-node third order edge
                            # (2 nodes at vertices, 2 internal to edge)
        elm_type[27] = 5    # 5-node fourth order edge
                            # (2 nodes at vertices, 3 internal to edge)
        elm_type[28] = 6    # 6-node fifth order edge
                            # (2 nodes at vertices, 4 internal to edge)
        elm_type[29] = 20   # 20-node third order tetrahedron
                            # (4 nodes at vertices, 12 with edges,
                            #  4 with faces)
        elm_type[30] = 35   # 35-node fourth order tetrahedron
                            # (4 nodes at vertices, 18 with edges,
                            #  12 with faces, 1 in volume)
        elm_type[31] = 56   # 56-node fifth order tetrahedron
                            # (4 nodes at vertices, 24 with edges,
                            #  24 with faces, 4 in volume)
        self.elm_type = elm_type

    def refine2dtri(self, marked_elements=None):
        """
        marked_elements : array
            list of marked elements for refinement.  None means uniform.
        bdy_ids : array
            list of ids for boundary lists
        """
        E = self.Elmts[2][1]
        Nel = E.shape[0]
        Nv = self.Verts.shape[0]

        if marked_elements is None:
            marked_elements = numpy.arange(0, Nel)

        marked_elements = numpy.ravel(marked_elements)
        #################################################################
        # construct vertex to vertex graph
        col = E.ravel()
        row = numpy.kron(numpy.arange(0, Nel), [1, 1, 1])
        data = numpy.ones((Nel*3,))
        V2V = coo_matrix((data, (row, col)), shape=(Nel, Nv))
        V2V = V2V.T * V2V

        # compute interior edges list
        V2VFullUpper = triu(V2V, 1).tocoo()
        Nint = 0
        V2V.data = numpy.ones(V2V.data.shape)
        V2Vupper = triu(V2V, 1).tocoo()

        # construct EdgeList from V2V
        Nedges = len(V2Vupper.data)
        V2Vupper.data = numpy.arange(0, Nedges)
        EdgeList = numpy.vstack((V2Vupper.row, V2Vupper.col)).T
        self.EdgeList = EdgeList
        Nedges = EdgeList.shape[0]

        # elements to edge list
        V2Vupper = V2Vupper.tocsr()
        edges = numpy.vstack((E[:, [0, 1]],
                             E[:, [1, 2]],
                             E[:, [2, 0]]))
        edges.sort(axis=1)
        ElementToEdge = V2Vupper[edges[:, 0], edges[:, 1]].reshape((3, Nel)).T
        self.ElementToEdge = ElementToEdge

        # mark edges as boundary
        BE = self.Elmts[1][1]
        Bid = self.Elmts[1][0]
        BE.sort(axis=1)
        BEdgeList = numpy.zeros((BE.shape[0],), dtype=int)
        i = 0
        for ed in BE:
            ed.sort()
            id0 = numpy.where(EdgeList[:, 0] == ed[0])[0]
            id1 = numpy.where(EdgeList[:, 1] == ed[1])[0]
            id = numpy.intersect1d(id0, id1)
            if len(id) == 1:
                id = id[0]
                BEdgeList[i] = id
                i += 1
        BEdgeFlag = numpy.zeros((Nedges,), dtype=bool)
        BEdgeFlag[BEdgeList] = True
        ########################################################

        marked_edges = numpy.zeros((Nedges,), dtype=bool)
        marked_edges[ElementToEdge[marked_elements, :].ravel()] = True

        # mark 3-2-1 triangles
        nsplit = len(numpy.where(marked_edges is True)[0])
        edge_num = marked_edges[ElementToEdge].sum(axis=1)
        edges3 = numpy.where(edge_num >= 2)[0]
        #edges3 = marked_edges[id]             # all 2 or 3 edge elements
        marked_edges[ElementToEdge[edges3, :]] = True  # marked 3rd edge
        #nsplit = len(numpy.where(marked_edges == True)[0]) - nsplit

        edges1 = numpy.where(edge_num == 1)[0]
        #edges1 = edge_num[id]             # all 2 or 3 edge elements

        # new nodes (only edges3 elements)

        x_new = 0.5*(self.Verts[EdgeList[marked_edges, 0], 0]) \
            + 0.5*(self.Verts[EdgeList[marked_edges, 1], 0])
        y_new = 0.5*(self.Verts[EdgeList[marked_edges, 0], 1]) \
            + 0.5*(self.Verts[EdgeList[marked_edges, 1], 1])
        z_new = 0.5*(self.Verts[EdgeList[marked_edges, 0], 2]) \
            + 0.5*(self.Verts[EdgeList[marked_edges, 1], 2])

        Verts_new = numpy.vstack((x_new, y_new, z_new)).T
        self.Verts = numpy.vstack((self.Verts, Verts_new))
        # indices of the new nodes
        new_id = numpy.zeros((Nedges,), dtype=int)
        new_id[marked_edges] = Nv + numpy.arange(0, nsplit)
        # New tri's in the case of refining 3 edges
        # example, 1 element
        #                n2
        #               / |
        #             /   |
        #           /     |
        #        n5-------n4
        #       / \      /|
        #     /    \    / |
        #   /       \  /  |
        # n0 --------n3-- n1
        ids = numpy.ones((Nel,), dtype=bool)
        ids[edges3] = False
        ids[edges1] = False
        id2 = numpy.where(ids is True)[0]

        E_new = numpy.delete(E, marked_elements, axis=0)  # E[id2, :]
        n0 = E[edges3, 0]
        n1 = E[edges3, 1]
        n2 = E[edges3, 2]
        n3 = new_id[ElementToEdge[edges3, 0]].ravel()
        n4 = new_id[ElementToEdge[edges3, 1]].ravel()
        n5 = new_id[ElementToEdge[edges3, 2]].ravel()

        t1 = numpy.vstack((n0, n3, n5)).T
        t2 = numpy.vstack((n3, n1, n4)).T
        t3 = numpy.vstack((n4, n2, n5)).T
        t4 = numpy.vstack((n3, n4, n5)).T

        E_new = numpy.vstack((E_new, t1, t2, t3, t4))
        self.Elmts[2] = (0, E_new)

    def smooth2dtri(self, maxit=10, tol=0.01):
        edge0 = self.Elmts[2][1][:, [0, 0, 1, 1, 2, 2]].ravel()
        edge1 = self.Elmts[2][1][:, [1, 2, 0, 2, 0, 1]].ravel()
        nedges = edge0.shape[0]
        data = numpy.ones((nedges,), dtype=int)
        #S = sparse(mesh.tri(:, [1, 1, 2, 2, 3, 3]),
        # mesh.tri(:, [2, 3, 1, 3, 1, 2]), 1, mesh.n, mesh.n);
        S = coo_matrix((data, (edge0, edge1)), shape=(self.Verts.shape[0],
                       self.Verts.shape[0])).tocsr().tocoo()
        S0 = S.copy()
        S.data = 0*S.data + 1

        W = S.sum(axis=1).ravel()

        L = (self.Verts[edge0, 0] - self.Verts[edge1, 0])**2 + \
            (self.Verts[edge0, 1] - self.Verts[edge1, 1])**2

        L_to_low = numpy.where(L < 1e-14)[0]
        L[L_to_low] = 1e-14

        # find the boundary nodes for this mesh (does not support a one-element
        # whole)
        bid = numpy.where(S0.data == 1)[0]
        bid = numpy.unique1d(S0.row[bid])
        self.bid = bid

        for iter in range(0, maxit):
            x_new = numpy.array(S*self.Verts[:, 0] / W).ravel()
            y_new = numpy.array(S*self.Verts[:, 1] / W).ravel()
            x_new[bid] = self.Verts[bid, 0]
            y_new[bid] = self.Verts[bid, 1]
            self.Verts[:, 0] = x_new
            self.Verts[:, 1] = y_new
            L_new = (self.Verts[edge0, 0] - self.Verts[edge1, 0])**2 + \
                    (self.Verts[edge0, 1] - self.Verts[edge1, 1])**2
            L_to_low = numpy.where(L < 1e-14)[0]
            L_new[L_to_low] = 1e-14

            move = max(abs((L_new-L) / L_new))  # inf norm

            if move < tol:
                return
            L = L_new

if __name__ == '__main__':

    #meshname = 'test.msh'
    meshname = 'bagel.msh'
    mesh = Mesh()
    mesh.read_msh(meshname)
    mesh.refine2dtri()
    mesh.refine2dtri()
    mesh.refine2dtri()
    mesh.smooth2dtri()
    print mesh.Elmts[2][1].shape

    import trimesh
    trimesh.trimesh(mesh.Verts[:, :2], mesh.Elmts[2][1])
    import pylab
    pylab.plot(mesh.Verts[mesh.bid, 0], mesh.Verts[mesh.bid, 1], 'ro')
    if 0:
        import trimesh
        trimesh.trimesh(mesh.Verts[:, :2], mesh.Elmts[2][1])

        mesh.refine2dtri([0, 3, 5])
        trimesh.trimesh(mesh.Verts[:, :2], mesh.Elmts[2][1])

        mesh.refine2dtri()
        mesh.refine2dtri()
        mesh.refine2dtri()
        trimesh.trimesh(mesh.Verts[:, :2], mesh.Elmts[2][1])
        print mesh.Elmts[2][1].shape

    #mesh.refine2dtri(marked_elements=[0, 3, 5])

    import pylab
    import demo
    #demo.trimesh(mesh.Verts[:, 0:2],mesh.Elmts[2][1])

  # mesh.write_vtu()
    #mesh.write_neu(bdy_ids=4)
  #  mesh.write_neu()
