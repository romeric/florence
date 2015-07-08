#include <IGESControl_Reader.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Edge.hxx>
#include <BRep_Tool.hxx>
#include <Geom_BSplineCurve.hxx>
#include <NCollection_Handle.hxx>

void main()
{
char *my_IGES_file="home/roman/Dropbox/2015_HighOrderMeshing/examples/rae2822.igs";
IGESControl_Reader reader;
IFSelect_ReturnStatus stat = reader.ReadFile(my_IGES_file);
Standard_Boolean failsonly = Standard_False; 
IFSelect_PrintCount mode = IFSelect_EntitiesByItem;
reader.PrintCheckLoad(failsonly,mode);

Handle(TColStd_HSequenceOfTransient) list = reader.GiveList("iges-basic-curves-3d");
     
Standard_Integer nbtrans = reader.TransferList (list);

TopoDS_Shape shape = reader.OneShape();
reader.PrintTransferInfo(IFSelect_FailOnly,IFSelect_Mapping);

TopoDS_Edge edge = TopoDS::Edge(shape);

TopLoc_Location CurveLocation;
Standard_Real CurveStart, CurveEnd;
Handle(Geom_Curve) curve = BRep_Tool::Curve(edge, CurveLocation, CurveStart, CurveEnd);

Handle(Geom_BSplineCurve) BsCurve = Handle(Geom_BSplineCurve)::DownCast(curve);
}