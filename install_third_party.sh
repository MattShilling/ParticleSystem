#! /bin/bash

mkdir -p third_party/ 
cd third_party
git clone https://github.com/libglui/glui.git 2>/dev/null 
mkdir -p build
cd glui
cmake -DCMAKE_INSTALL_PREFIX:PATH=third_party/build
make install -j4

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:third_party/build/lib
export LD_LIBRARY_PATH