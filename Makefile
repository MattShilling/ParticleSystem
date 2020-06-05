objects = sample.o

CXX = g++

all: $(objects)
	$(CXX) $(objects) /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 /usr/local/apps/cuda/cuda-10.1/lib64/libcudart.so.10.1.243 -lGL -lm -lglut -fopenmp -lGLU third_party/build/lib/libglui.so -o sample

%.o: %.cpp
	$(CXX) -Ithird_party/build/include -I/usr/local/apps/cuda/cuda-10.1/include/ -c $< -o $@ -std=c++11

clean:
	rm -f *.o sample

format:
	clang-format-3.9 -i *.cpp *.h *.cl
