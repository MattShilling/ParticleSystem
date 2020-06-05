#! /bin/bash

mkdir -p third_party/ 
cd third_party
git clone https://github.com/libglui/glui.git 2>/dev/null
mkdir -p build
cd glui
cmake -DCMAKE_INSTALL_PREFIX:PATH=~/code/cs475/ParticleSystem/third_party/build
make install -j4

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/code/cs475/ParticleSystem/third_party/build/lib
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/apps/cuda/cuda-10.1/lib64/
export LD_LIBRARY_PATH