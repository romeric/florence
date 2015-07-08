TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += c++11

SOURCES += main.cpp \
    occ_frontend.cpp \
    py_to_occ_frontend.cpp

QMAKE_CXXFLAGS += -Wno-unused #-O2 -msse -msse2 -fomit-frame-pointer -fno-strict-aliasing

INCLUDEPATH +=/home/roman/Dropbox/Eigen
#INCLUDEPATH +=/usr/local/inc
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
    cnp_funcs.hpp \
    occ_inc.hpp \
    cnp_funcs.hpp \
    aux_funcs.hpp \
    eigen_inc.hpp \
    occ_frontend.hpp \
    std_inc.hpp \
    py_to_occ_frontend.hpp

