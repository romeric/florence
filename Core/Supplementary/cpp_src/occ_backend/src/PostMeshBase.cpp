
#include <PostMeshBase.hpp>


PostMeshBase::PostMeshBase()
{
}

PostMeshBase::~PostMeshBase()
{
}


//void PostMeshBase::ReadIGES(const char* filename)
//{
//    //! IGES FILE READER BASED ON OCC BACKEND
//    //! THIS FUNCTION CAN BE EXPANDED FURTHER TO TAKE CURVE/SURFACE CONSISTENY INTO ACCOUNT
//    //! http://www.opencascade.org/doc/occt-6.7.0/overview/html/user_guides__iges.html

//    IGESControl_Reader reader;
//    //IFSelect_ReturnStatus stat  = reader.ReadFile(filename.c_str());
//    reader.ReadFile(filename);
//    // CHECK FOR IMPORT STATUS
//    reader.PrintCheckLoad(Standard_True,IFSelect_GeneralInfo);
//    reader.PrintCheckTransfer(Standard_True,IFSelect_ItemsByEntity);
//    // READ IGES FILE AS-IS
//    Interface_Static::SetIVal("read.iges.bspline.continuity",0);
//    Standard_Integer ic =  Interface_Static::IVal("read.iges.bspline.continuity");
//    if (ic !=0)
//    {
//        std::cout << "IGES file was not read as-is. The file was not read/transformed correctly";
//    }

//    //Interface_Static::SetIVal("xstep.cascade.unit",0);
//    //Interface_Static::SetIVal("read.scale.unit",0);

//    // IF ALL OKAY, THEN TRANSFER ROOTS
//    reader.TransferRoots();

//    this->imported_shape  = reader.OneShape();
//    this->no_of_shapes = reader.NbShapes();

////    print(no_of_shapes);
////    exit(EXIT_FAILURE);

////        Handle_TColStd_HSequenceOfTransient edges = reader.GiveList("iges-faces");


//}
