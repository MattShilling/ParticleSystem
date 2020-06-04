# Particle System Simulator 

## Requirements

- OpenGL
- OpenCL (`cl_khr_gl_sharing` extension required)
- GLUI (https://github.com/libglui/glui) 
    - `install_third_party.sh` should take care of this for you. See: **Building**.
- GLUT
- OpenMP (will remove dependency)

## Bulding

- Make sure that third party components are installed with:
    - `install_third_party.sh`

- Makefile
    - `make`: Build the project
    - `make clean`: Clean up after yourself!
    - `make format`: Make code files nice and neat.

