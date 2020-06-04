objects = sample.o

CXX = g++

all: $(objects)
	$(CXX) $(objects) -lOpenCL -lGL -lm -lglut -fopenmp -lGLU third_party/build/lib/libglui.so -o sample

%.o: %.cpp
	$(CXX) -Ithird_party/build/include -c $< -o $@ -std=c++11

clean:
	rm -f *.o sample

format:
	clang-format-3.9 -i *.cpp *.h *.cl
