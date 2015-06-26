from OCC.TopoDS import *
from OCC.TopExp import TopExp_Explorer
from OCC.TopAbs import TopAbs_VERTEX, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
import OCC.TopoDS as TopoDS

# from OCC.gp import *
# from OCC.Precision import *
# from OCC.BRepBuilderAPI import *
# # import OCC.Utils.DataExchange.STEP
# from OCC.BRepPrimAPI import *
# from OCC.TopoDS import *
  
# #for BoundingBox
# from OCC.Bnd import *
# from OCC.BRepBndLib import *
  
# #for exploring shapes
# from OCC.BRep import BRep_Tool
# from OCC.BRepTools import BRepTools_WireExplorer
# # from OCC.Utils.Topology import WireExplorer #wraps BRepTools_WireExplorer
# from OCC.TopExp import TopExp_Explorer
# from OCC.TopAbs import TopAbs_VERTEX, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
  
# # from skdb import Connection, Part, Interface, Unit, FennObject, prettyfloat
# import os, math
# from copy import copy, deepcopy
# from string import Template

#where should this go?
def extend(first, second, overwrite=False):
    '''adds items in the second dictionary to the first
    overwrite=True means that the second dictionary is going to win in the case of conflicts'''
    third = first.copy()
    for second_key in second.keys():
        if first.get(second_key) is not None and overwrite:
            third[second_key] = copy(second[second_key])
        elif first.get(second_key) is None:
            third[second_key] = copy(second[second_key])
    return third

class OCC_shape(TopoDS_Shape):
    occ_class = TopoDS_Shape #just a default
    def __init__(self, incoming=None):
        if isinstance(incoming, self.__class__): #Face(Face()), Shape(Shape()), etc.
            #if self.occ_class is not TopoDS_Shape:
            TopoDS_Shape.__init__(self)
            self.occ_class.__init__(self) #yes, no? dunno
            self.__dict__ = extend(self.__dict__, incoming.__dict__, overwrite=False)
            self.TShape(incoming.TShape())
        elif isinstance(incoming, self.occ_class):
            #if self.occ_class is not TopoDS_Shape:
            TopoDS_Shape.__init__(self)
            self.occ_class.__init__(self)
            self.__dict__ = extend(self.__dict__, incoming.__dict__, overwrite=False)
            other_shape = incoming.TShape()
            self.TShape(other_shape)
        elif isinstance(incoming, TopoDS_Shape):
            TopoDS_Shape.__init__(self)
            self.occ_class.__init__(self)
            self.__dict__ = extend(self.__dict__, incoming.__dict__, overwrite=False)
            other_shape = incoming.TShape()
            self.TShape(other_shape)
    def __copy__(self):
        '''copies only the shape and returns a Shape with the shape'''
        #note that occ uses TopoDS_Shape().TShape() for some reason.
        current_shape = copy(self.TShape())
        new_shape = Shape()
        new_shape.TShape(current_shape)
        return new_shape
    def get_wires(self):
        '''traverses a face (or a shape) for wires
        you might want get_edges instead?
        returns a list of wires'''
        wires = []
        #if isinstance(self, Shape): to_explore = self
        #else: to_explore = Shape(self)
        to_explore = self
        explorer = TopExp_Explorer(to_explore, TopAbs_WIRE, TopAbs_EDGE)
        explorer.ReInit()
        while explorer.More():
            #edge = TopoDS().Edge(explorer.Current())
            #wire = BRepBuilderAPI_MakeWire(edge).Wire()
            wire = TopoDS().Wire(explorer.Current())
            wire1 = Wire(wire)
            wires.append(wire1)
            explorer.Next()
        return wires
    def get_edges(self):
        '''sometimes what you really want is get_edges instead of get_wires. (a wire will not have points, but an edge will)
        can someone make sense of this please?'''
        edges = []
        explorer = TopExp_Explorer(self, TopAbs_EDGE)
        explorer.ReInit()
        while explorer.More():
            edge = TopoDS().Edge(explorer.Current())
            edge1 = Edge(edge)
            edges.append(edge1)
            explorer.Next()
        return edges
    def get_faces(self):
        '''returns a list of faces for the given shape
        note that it could be that the entire shape is really just a face, so don't be upset if this returns []'''
        faces = []
        explorer = TopExp_Explorer(self, TopAbs_FACE)
        while explorer.More():
            face = Face(TopoDS().Face(explorer.Current()))
            faces.append(face)
            explorer.Next()
        return faces
    def get_points(self, edges=True, wires=True):
        '''returns a list of points defining the shape
        based off of wires and faces'''
        points = []
        #wires = self.wires
        faces = self.faces
        #for wire in wires:
        #    print "get_points: processing a wire"
        #    points.extend(wire.points)
        for face in faces:
            points.extend(face.get_points(edges=edges,wires=wires))
        points = list(set(points)) #filter out the repeats
        return points
    def get_shape(self):
        '''returns the TShape object
        occ only, this probably isn't useful anywhere else'''
        return self.TShape()
    def set_shape(self, value):
        '''sets the TShape'''
        if isinstance(value, Handle_TopoDS_TShape):
            self.TShape(value)
        elif isinstance(value, self.__class__):
            self.TShape(value.TShape())
        elif isinstance(value, self.occ_class):
            self.TShape(value.TShape())
        elif isinstance(value, self.occ_class_handler):
            #set_shape(Handle_TopoDS_TFace)
            self.TShape(value.GetObject().TShape())
        else: raise ValueError, "OCC_shape.set_shape: not sure what to do, possibilities exhausted for %s" % (value)
    wires = property(fset=None, fget=get_wires, doc="a list of wires (partially) defining the shape")
    edges = property(fset=None, fget=get_edges, doc="a list of edges (partially) defining the shape")
    faces = property(fset=None, fget=get_faces, doc="a list of faces (partially) defining the shape")
    points = property(fset=None, fget=get_points, doc="a list of points (partially) defining the shape")
    shape = property(fset=set_shape, fget=get_shape, doc="wraps occ TopoDS_Shape TShape")










from OCC.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge,
                                BRepBuilderAPI_MakeVertex,
                                BRepBuilderAPI_MakeWire)
from OCC.BRepFill import BRepFill_Filling
from OCC.GeomAbs import GeomAbs_C0
from OCC.GeomAPI import GeomAPI_PointsToBSpline
from OCC.TColgp import TColgp_Array1OfPnt


def make_edge(*args):
    edge = BRepBuilderAPI_MakeEdge(*args)
    result = edge.Edge()
    return result


def make_vertex(*args):
    vert = BRepBuilderAPI_MakeVertex(*args)
    result = vert.Vertex()
    return result


def make_n_sided(edges, continuity=GeomAbs_C0):
    n_sided = BRepFill_Filling()  # TODO Checck optional NbIter=6)
    for edg in edges:
        n_sided.Add(edg, continuity)
    n_sided.Build()
    face = n_sided.Face()
    return face


def make_wire(*args):
    # if we get an iterable, than add all edges to wire builder
    if isinstance(args[0], list) or isinstance(args[0], tuple):
        wire = BRepBuilderAPI_MakeWire()
        for i in args[0]:
            wire.Add(i)
        wire.Build()
        return wire.Wire()
    wire = BRepBuilderAPI_MakeWire(*args)
    return wire.Wire()


def points_to_bspline(pnts):
    pts = TColgp_Array1OfPnt(0, len(pnts)-1)
    for n, i in enumerate(pnts):
        pts.SetValue(n, i)
    crv = GeomAPI_PointsToBSpline(pts)
    return crv.Curve()
























import urllib2

from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Geom2dAPI import Geom2dAPI_PointsToBSpline
from OCC.GeomAPI import geomapi
from OCC.gp import gp_Pnt, gp_Vec, gp_Pnt2d, gp_Pln, gp_Dir
from OCC.TColgp import TColgp_Array1OfPnt2d

# from core_geometry_utils import make_wire, make_edge


class UiucAirfoil:
    """
    Airfoil with a section from the UIUC database
    """
    def __init__(self, chord, span, profile):
        self.chord = chord
        self.span = span
        self.profile = profile
        self.shape = self.make_shape()

    def make_shape(self):
        # 1 - retrieve the data from the UIUC airfoil data page
        foil_dat_url = 'http://www.ae.illinois.edu/m-selig/ads/coord_seligFmt/%s.dat' % self.profile
        print foil_dat_url
        f = urllib2.urlopen(foil_dat_url)

        plan = gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.))  # Z=0 plan / XY plan
        section_pts_2d = []

        for line in f.readlines()[1:]:  # The first line contains info only
            # 2 - do some cleanup on the data (mostly dealing with spaces)
            line = line.lstrip().rstrip().replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')
            data = line.split(' ')  # data[0] = x coord.    data[1] = y coord.

            # 3 - create an array of points
            if len(data) == 2:  # two coordinates for each point
                section_pts_2d.append(gp_Pnt2d(float(data[0])*self.chord,
                                               float(data[1])*self.chord))
        print section_pts_2d[0].X()
        # 4 - use the array to create a spline describing the airfoil section
        spline_2d = Geom2dAPI_PointsToBSpline(point2d_list_to_TColgp_Array1OfPnt2d(section_pts_2d),
                                              len(section_pts_2d)-1,  # order min
                                              len(section_pts_2d))   # order max
        spline = geomapi.To3d(spline_2d.Curve(), plan)
        # print dir(d)
        # 5 - figure out if the trailing edge has a thickness or not,
        # and create a Face
        try:
            #first and last point of spline -> trailing edge
            trailing_edge = make_edge(gp_Pnt(section_pts_2d[0].X(), section_pts_2d[0].Y(), 0.0),
                                      gp_Pnt(section_pts_2d[-1].X(), section_pts_2d[-1].Y(), 0.0))
            face = BRepBuilderAPI_MakeFace(make_wire([make_edge(spline), trailing_edge]))
        except AssertionError:
            # the trailing edge segment could not be created, probably because
            # the points are too close
            # No need to build a trailing edge
            face = BRepBuilderAPI_MakeFace(make_wire(make_edge(spline)))

        # 6 - extrude the Face to create a Solid
        return BRepPrimAPI_MakePrism(face.Face(),
                                     gp_Vec(gp_Pnt(0., 0., 0.),
                                     gp_Pnt(0., 0., self.span))).Shape()


def point2d_list_to_TColgp_Array1OfPnt2d(li):
    """
    List of gp_Pnt2d to TColgp_Array1OfPnt2d
    """
    return _Tcol_dim_1(li, TColgp_Array1OfPnt2d)


def _Tcol_dim_1(li, _type):
    """
    Function factory for 1-dimensional TCol* types
    """
    pts = _type(0, len(li)-1)
    for n, i in enumerate(li):
        pts.SetValue(n, i)
    return pts

if __name__ == '__main__':
    from OCC.Display.SimpleGui import init_display
    # display, start_display, add_menu, add_function_to_menu = init_display()
    airfoil = UiucAirfoil(50., 200., 'b737a')
    # print airfoil
    # display.DisplayShape(airfoil.shape, update=True)
    # start_display()