TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += c++11

SOURCES += main.cpp \
    OCCPlugin.cpp \
    OCCPluginInterface.cpp

QMAKE_CXXFLAGS += -Wno-unused -std=c++11 # -O2 #-fopenmp -msse -msse2 -fomit-frame-pointer -fno-strict-aliasing

INCLUDEPATH +=/home/roman/Dropbox/eigen-devel
#INCLUDEPATH +=/home/roman/Dropbox/eigen
INCLUDEPATH +=/usr/local/include/oce/

LIBS += -L/usr/local/lib -l:libTKIGES.so.9 -l:libTKXSBase.so.9 -l:libTKBRep.so.9 -l:libTKernel.so.9 -l:libTKTopAlgo.so.9 \
     -l:libTKGeomBase.so.9 -l:libTKMath.so.9 -l:libTKHLR.so.9 -l:libTKG2d.so.9 -l:libTKBool.so.9 -l:libTKG3d.so.9 -l:libTKOffset.so.9 \
     -l:libTKG2d.so.9 -l:libTKXMesh.so.9 -l:libTKGeomAlgo.so.9 -l:libTKShHealing.so.9 -l:libTKFeat.so.9 -l:libTKFillet.so.9 \
     -l:libTKBO.so.9 -l:libTKPrim.so.9

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


include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    EIGEN_INC.hpp \
    OCCPlugin.hpp \
    AuxFuncs.hpp \
    CNPFuncs.hpp \
    OCC_INC.hpp \
    STL_INC.hpp \
    OCCPluginInterface.hpp

