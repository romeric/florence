TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += c++11

SOURCES += src/main.cpp \
    src/PostMeshBase.cpp \
    src/PostMeshCurve.cpp \
    src/PostMeshSurface.cpp \
    src/Examples.cpp

#QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -pipe# -Wdelete-non-virtual-dtor -Wno-unused # -O2 #-fopenmp -msse -msse2 -fomit-frame-pointer -fno-strict-aliasing
QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -Wno-unused -D_OCC64#-O3 -fopenmp


#INCLUDEPATH += include/
#INCLUDEPATH +=/home/roman/Dropbox/eigen-devel
INCLUDEPATH +=/usr/local/include/eigen/
INCLUDEPATH +=/usr/local/include/oce/


LIBS += -L/usr/local/lib -l:libTKIGES.so.9 -l:libTKSTEP.so.9 -l:libTKXSBase.so.9 -l:libTKBRep.so.9 -l:libTKernel.so.9 -l:libTKTopAlgo.so.9 \
     -l:libTKGeomBase.so.9 -l:libTKMath.so.9 -l:libTKHLR.so.9 -l:libTKG2d.so.9 -l:libTKBool.so.9 -l:libTKG3d.so.9 -l:libTKOffset.so.9 \
     -l:libTKXMesh.so.9 -l:libTKMesh.so.9 -l:libTKMeshVS.so.9 -l:libTKGeomAlgo.so.9 -l:libTKShHealing.so.9 -l:libTKFeat.so.9 -l:libTKFillet.so.9 \
     -l:libTKBO.so.9 -l:libTKPrim.so.9 -l:libTKAdvTools.so -l:libTKPShape.so -l:libTKBO.so.9 -l:libTKXSBase.so.9 -l:libTKTopAlgo.so.9

#LIBS += -L/usr/lib/gcc/x86_64-linux-gnu/4.8.4 -lgomp

#LIBS += -L/usr/local/lib -lTKIGES -lTKXSBase -lTKBRep -lTKernel -lTKTopAlgo -lTKGeomBase -lTKMath -lTKHLR -lTKG2d -lTKBool \
#    -lTKXMesh\
#    -lTKFillet\
#    -lTKGeomBase\
#    -lTKPrim\
#    -lTKOffset\
#    -lTKHLR\
#    -lTKMath\
#    -lTKBO\
#    -lTKG2d\
#    -lTKG3d\
#    -lTKShHealing\
#    -lTKBRep\
#    -lTKBool\
#    -lTKBRep\
#    -lTKTopAlgo\
#    -lTKMesh\
#    -lTKFeat\
#    -lTKGeomAlgo



HEADERS += \
    AuxFuncs.hpp \
    CNPFuncs.hpp \
    EIGEN_INC.hpp \
    OCC_INC.hpp \
    PostMeshBase.hpp \
    STL_INC.hpp \
    PostMeshCurve.hpp \
    PostMeshSurface.hpp \
    PyInterfaceEmulator.hpp \
    PyInterface.hpp
#    include/OCCPlugin.hpp \
#    include/AuxFuncs.hpp \
#    include/CNPFuncs.hpp \
#    include/EIGEN_INC.hpp \
#    include/OCC_INC.hpp \
#    include/OCCPluginInterface.hpp \
#    include/STL_INC.hpp \
#    include/PostMeshBase.hpp \

