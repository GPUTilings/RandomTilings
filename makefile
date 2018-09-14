CXX = g++
LDLIBS = OpenCL
LDFLAGS = "C:\Program Files (x86)\Intel\OpenCL SDK\6.3\lib\x86"
srcdir = "C:\Program Files (x86)\Intel\OpenCL SDK\6.3\include"
CXXFLAGS = -w -std=gnu++11

opencl_test: common.o src/common/common.h
	$(CXX) $(CXXFLAGS) -o opencl_test common.o src/opencl_test.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)

MinimalDominoEx: common.o src/common/common.h file_reader.o DominoTiler.o src/DominoTiler/DominoTiler.h 
	$(CXX) $(CXXFLAGS) -o MinimalDominoEx common.o file_reader.o DominoTiler.o src/MinimalDominoEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
	
AztecDiamondCFTPEx: common.o src/common/common.h file_reader.o DominoTiler.o src/DominoTiler/DominoTiler.h 
	$(CXX) $(CXXFLAGS) -o AztecDiamondCFTPEx common.o file_reader.o DominoTiler.o src/AztecDiamondCFTPEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	
	
RectTriangleEx: common.o src/common/common.h file_reader.o RectTriangleTiler.o src/RectTriangle/RectTriangleTiler.h
	$(CXX) $(CXXFLAGS) -o RectTriangleEx common.o file_reader.o RectTriangleTiler.o src/RectTriangleEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	
	
TriangleDimerEx: common.o src/common/common.h file_reader.o TriangleDimerTiler.o src/TriangleDimer/TriangleDimerTiler.h
	$(CXX) $(CXXFLAGS) -o TriangleDimerEx common.o file_reader.o TriangleDimerTiler.o src/TriangleDimerEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	

MinimalLozengeEx: common.o src/common/common.h file_reader.o LozengeTiler.o src/Lozenge/LozengeTiler.h
	$(CXX) $(CXXFLAGS) -o MinimalLozengeEx common.o file_reader.o LozengeTiler.o src/MinimalLozengeEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	

common.o: src/common/common.cpp src/common/common.h
	$(CXX) $(CXXFLAGS) -c src/common/common.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
	
DominoTiler.o: src/Domino/DominoTiler.cpp src/Domino/DominoTiler.h src/common/common.h src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/Domino/DominoTiler.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
	
LozengeTiler.o: src/Lozenge/LozengeTiler.cpp src/Lozenge/LozengeTiler.h src/common/common.h src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/Lozenge/LozengeTiler.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	

RectTriangleTiler.o: src/RectTriangle/RectTriangleTiler.cpp src/RectTriangle/RectTriangleTiler.h src/common/common.h src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/RectTriangle/RectTriangleTiler.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
	
TriangleDimerTiler.o: src/TriangleDimer/TriangleDimerTiler.cpp src/TriangleDimer/TriangleDimerTiler.h src/common/common.h src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/TriangleDimer/TriangleDimerTiler.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)

file_reader.o: src/TinyMT/file_reader.cpp src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/TinyMT/file_reader.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)

clean:
	rm *.o