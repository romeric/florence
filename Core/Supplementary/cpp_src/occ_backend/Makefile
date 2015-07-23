CXX = g++
INCDIRS = -I/home/roman/Dropbox/eigen-devel/ -I/usr/local/include/oce/ -I.
LIBS = -L/usr/local/lib -l:libTKIGES.so.9 -l:libTKXSBase.so.9 -l:libTKBRep.so.9 -l:libTKernel.so.9 -l:libTKTopAlgo.so.9 \
	   -l:libTKGeomBase.so.9 -l:libTKMath.so.9 -l:libTKHLR.so.9 -l:libTKG2d.so.9 -l:libTKBool.so.9 -l:libTKG3d.so.9 -l:libTKOffset.so.9 \
	   -l:libTKG2d.so.9 -l:libTKXMesh.so.9 -l:libTKGeomAlgo.so.9 -l:libTKShHealing.so.9 -l:libTKFeat.so.9 -l:libTKFillet.so.9 \
	   -l:libTKBO.so.9 -l:libTKPrim.so.9
OPTFLAGS = -O0
CPPFLAGS = -std=c++11 -Wno-unused -pipe

TARGET = main
OBJ1 = main.o
OBJ2 = py_to_occ_frontend.o
OBJ3 = occ_frontend.o
SRC1 = main.cpp
SRC3 = occ_frontend.cpp
SRC2 = py_to_occ_frontend.cpp


all:$(TARGET)

$(TARGET):$(OBJ1) $(OBJ2) $(OBJ3)
   $(CXX) $(CPPFLAGS) $(INCDIRS) $(LIBS) $(OBJ1) $(OBJ2) $(OBJ3) -o $(TARGET)
$(OBJ1): $(SRC1) 
	$(CXX) $(CPPFLAGS) $(INCDIRS) $(SRC1) -c  $(OBJ1) 
$(OBJ2): $(SRC2) 
	$(CXX) $(CPPFLAGS) $(INCDIRS) $(SRC2) -c  $(OBJ2) 
$(OBJ3): $(SRC3) 
	$(CXX) $(CPPFLAGS) $(INCDIRS) $(SRC3) -c  $(OBJ3) 
