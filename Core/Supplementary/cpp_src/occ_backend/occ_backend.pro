TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += c++11

SOURCES += main.cpp

QMAKE_CXXFLAGS += -Wno-unused #-O2

INCLUDEPATH +=/home/roman/Dropbox/Eigen
#INCLUDEPATH +=/usr/local/inc
#INCLUDEPATH +=/home/roman/Downloads/opencascade-6.9.0/inc/
INCLUDEPATH +=/usr/local/include/oce/

LIBS += -L/usr/local/lib -l:libTKIGES.so.9 -l:libTKXSBase.so.9 -l:libTKBRep.so.9 -l:libTKernel.so.9 -l:libTKTopAlgo.so.9 \
     -l:libTKGeomBase.so.9 -l:libTKMath.so.9 -l:libTKHLR.so.9 -l:libTKG2d.so.9 -l:libTKBool.so.9 -l:libTKG3d.so.9 -l:libTKOffset.so.9 \
     -l:libTKG2d.so.9 -l:libTKMath.so.9 -l:libTKXMesh.so.9 -l:libTKGeomAlgo.so.9 #"oce-0.16"

#LIBS += -L/usr/local/lib -lTKIGES -lTKXSBase -lTKBRep -lTKernel -lTKTopAlgo -lTKGeomBase -lTKMath -lTKHLR -lTKG2d -lTKBool \
#    -lTKG3d \
#    -lTKOffset\
#    -lTKShHealing\
#    -lTKXMesh\
#    -lTKFillet\
#    -lTKGeomBase\
#    -lTKTopAlgo\
#    -lTKG2d\
#    -lTKPrim\
#    -lTKOffset\
#    -lTKHLR\
#    -lTKMath\
#    -lTKFillet\
#    -lTKBO\
#    -lTKGeomBase\
#    -lTKG2d\
#    -lTKGeomAlgo\
#    -lTKG3d\
#    -lTKShHealing\
#    -lTKPrim\
#    -lTKBO\
#    -lTKBRep\
#    -lTKMesh\
#    -lTKBool\
#    -lTKBRep\
#    -lTKTopAlgo\
#    -lTKMesh\
#    -lTKFeat\
#    -lTKGeomAlgo

#LIBS += -L/usr/local/lib/libTKernel.so.9

include(deployment.pri)
qtcAddDeployment()

