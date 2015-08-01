#ifndef OCC_INC_HPP
#define OCC_INC_HPP

#include <Standard.hxx>
//#include <Geom_TrimmedCurve.hxx>
//#include <Geom2dAPI_ProjectPointOnCurve.hxx>
//#include <Geom2d_BSplineCurve.hxx>
//#include <Geom2d_TrimmedCurve.hxx>
//#include <TColgp_Array1OfPnt2d.hxx>
#include <IGESControl_Reader.hxx>
#include <IGESControl_Writer.hxx>
#include <XSControl_Reader.hxx>
#include <Interface_Static.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <Geom_BSplineCurve.hxx>
#include <GeomConvert.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <TopExp_Explorer.hxx>
#include <BRep_Tool.hxx>
#include <BRepBuilderAPI_NurbsConvert.hxx>
#include <gp_Pnt2d.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <Geom_Circle.hxx>
#include <gp.hxx>
#include <gp_Circ.hxx>
#include <Geom_Line.hxx>
#include <GeomAPI_ProjectPointOnCurve.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <GeomConvert.hxx>
#include <ShapeAnalysis_Curve.hxx>
#include <ShapeAnalysis_Surface.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <GeomConvert_CompCurveToBSplineCurve.hxx>
//#include <Hermit.hxx>


//! Identifies the type of a curve.
//! Not necessary to redefine as they are defined in OCC
//!
//enum GeomAbs_CurveType {
//    GeomAbs_Line,
//    GeomAbs_Circle,
//    GeomAbs_Ellipse,
//    GeomAbs_Hyperbola,
//    GeomAbs_Parabola,
//    GeomAbs_BezierCurve,
//    GeomAbs_BSplineCurve,
//    GeomAbs_OtherCurve
//};

//enum GeomAbs_SurfaceType {
//    GeomAbs_Plane,
//    GeomAbs_Cylinder,
//    GeomAbs_Cone,
//    GeomAbs_Sphere,
//    GeomAbs_Torus,
//    GeomAbs_BezierSurface,
//    GeomAbs_BSplineSurface,
//    GeomAbs_SurfaceOfRevolution,
//    GeomAbs_SurfaceOfExtrusion,
//    GeomAbs_OffsetSurface,
//    GeomAbs_OtherSurface
//}

#endif // OCC_INC_HPP

