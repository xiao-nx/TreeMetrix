CXX = g++
CXXFLAGS = -O2 -std=c++11 -fPIC -shared

HEADER_PATH = -I/usr/local/include/ \
        -I/usr/include/python2.7/ \
        -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ 

LIB_PATH = -L/usr/local/lib/ \
        -Wl,-rpath=/usr/local/lib/ \
 
LIBS = -lpython2.7 -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_legacy 

SRCS = $(wildcard ./*.cpp)
OBJS = $(patsubst %.cpp, %.o, $(SRCS))

TARGET = ../cmlswarp.so

$(TARGET) : $(OBJS)
	$(CXX)  $(CXXFLAGS) $^ -o $@ $(LIB_PATH) $(LIBS)

$(OBJS):%.o : %.cpp
	$(CXX)  $(CXXFLAGS) -c $< -o $@ $(HEADER_PATH)

clean:
	rm -rf $(TARGET) $(OBJS)
