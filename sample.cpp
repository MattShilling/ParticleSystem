#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>

// OpenGL includes.
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>

// GLUI include.
#include "GL/glui.h"

// OpenCL includes.
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include <cuda_runtime_api.h>

#include "cl_rig.h"

// opencl vendor ids:
#define ID_AMD 0x1002
#define ID_INTEL 0x8086
#define ID_NVIDIA 0x10de

// title of these windows:

const char *WINDOWTITLE = {
    "OpenCL/OpenGL Particle System -- Matt Shilling"};
const char *GLUITITLE = {"User Interface Window"};

// random parameters:

const float XMIN = {-100.0};
const float XMAX = {100.0};
const float YMIN = {-100.0};
const float YMAX = {100.0};
const float ZMIN = {-100.0};
const float ZMAX = {100.0};

const float VMIN = {-100.};
const float VMAX = {100.};

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32 // number of threads per block
#endif

const int NUM_PARTICLES = 1024 * 1024;

#define NUM_BLOCKS (NUM_PARTICLES / BLOCK_SIZE)

const char *CL_FILE_NAME = {"particles.cl"};
const char *CL_BINARY_NAME = {"particles.nv"};

const int GLUITRUE = {true};
const int GLUIFALSE = {false};

#define ESCAPE 0x1b

const int INIT_WINDOW_SIZE = {700};  // window size in pixels

const float ANGFACT = {1.};
const float SCLFACT = {0.005f};
const float MINSCALE = {0.001f};

const int LEFT = {4};
const int MIDDLE = {2};
const int RIGHT = {1};

enum Projections { ORTHO, PERSP };

enum ButtonVals { GO, PAUSE, RESET, QUIT };

const float BACKCOLOR[] = {0., 0., 0., 0.};

const GLfloat AXES_COLOR[] = {1., .5, 0.};
const GLfloat AXES_WIDTH = {3.};

// Structs we will need later.
struct xyzw {
    float x, y, z, w;
};

struct rgba {
    float r, g, b, a;
};

// Non-constant global variables.
int ActiveButton;  // current button that is down
GLuint AxesList;   // list to hold the axes
int AxesOn;        // ON or OFF
GLUI *Glui;        // instance of glui window
int GluiWindow;    // the glut id for the glui window
int MainWindow;    // window id for main graphics window
int Paused;
GLfloat RotMatrix[4][4];  // set by glui rotation widget
float Scale, Scale2;      // scaling factors
GLuint SphereList;
int WhichProjection;  // ORTHO or PERSP
int Xmouse, Ymouse;   // mouse values
float Xrot, Yrot;     // rotation angles in degrees
float TransXYZ[3];    // set by glui translation widgets

double ElapsedTime;
int ShowPerformance;

dim3 threads(BLOCK_SIZE, 1, 1);
dim3 grid(NUM_BLOCKS, 1, 1);

GLuint hPobj;
GLuint hCobj;
cudaGraphicsResource_t dPobj;
cudaGraphicsResource_t dCobj;
struct xyzw *hVel;
cl_mem dVel;
cl_command_queue CmdQueue;
cl_device_id Device;
cl_kernel Kernel;
cl_platform_id Platform;
cl_program Program;
cl_platform_id PlatformID;

// cudaGraphicsResource_t interop_buf_handle;

// Function prototypes.
inline float SQR(float x) { return x * x; }
float Ranf(float, float);
void Animate();
void Axes(float);
void Buttons(int);
void DrawScene();
void DoRasterString(float, float, float, char *);
void DoStrokeString(float, float, float, float, char *);
bool InitCL();
void InitGlui();
void InitGraphics();
void InitLists();
bool IsCLExtensionSupported(const char *);
void Keyboard(unsigned char, int, int);
void MouseButton(int, int, int, int);
void MouseMotion(int, int);
void PrintCLError(cl_int, const char * = "", FILE * = stderr);
void PrintOpenclInfo();
void Quit();
void Reset();
void ResetParticles();
void Resize(int, int);
void SelectOpenclDevice();
void Traces(int);
void Visibility(int);

#define GL_CHECK_ERROR() \
    do { \
        GLenum err; \
        err = glGetError(); \
        if(err != GL_NO_ERROR) \
        { \
            fprintf(stderr, "%s:%d GL error: %s\n", __FILE__, __LINE__, gluErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#define CUDA_CHECK_ERROR() \
    do { \
        cudaError err = cudaGetLastError(); \
        if (cudaSuccess != err) { \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
            exit(1); \
        } \
    } while(0) 

// Main Program.
int main(int argc, char *argv[]) {
    printf("%d\n\n\n", glXGetCurrentContext());
    printf("%d\n\n\n", glXGetCurrentDisplay());
    glutInit(&argc, argv);
    InitGraphics();
    InitLists();
    ASSERT(InitCL());
    Reset();
    InitGlui();
    glutMainLoop();
    return 0;
}

void Animate() {
    cl_int status;
    double time0, time1;

    // Acquire the vertex buffers from opengl.
    glutSetWindow(MainWindow);
    glFinish();

    status = clEnqueueAcquireGLObjects(CmdQueue, 1, &dPobj, 0,
                                       NULL, NULL);
    PrintCLError(status, "clEnqueueAcquireGLObjects (1): ");
    status = clEnqueueAcquireGLObjects(CmdQueue, 1, &dCobj, 0,
                                       NULL, NULL);
    PrintCLError(status, "clEnqueueAcquireGLObjects (2): ");

    if (ShowPerformance)
        time0 = omp_get_wtime();

    // Enqueue the Kernel object for execution.
    // cl_event wait;
    // status = clEnqueueNDRangeKernel(
    //     CmdQueue, Kernel, 1, NULL, GlobalWorkSize, LocalWorkSize,
    //     0, NULL, &wait);
    // PrintCLError(status, "clEnqueueNDRangeKernel: ");
    Kernel<<<grid, threads>>>(/** memory **/);

    if (ShowPerformance) {
        status = clWaitForEvents(1, &wait);
        PrintCLError(status, "clWaitForEvents: ");
        time1 = omp_get_wtime();
        ElapsedTime = time1 - time0;
    }

    cudaGraphicsUnregisterResource(dPobj);
    cudaGraphicsUnregisterResource(dCobj);

    // clFinish(CmdQueue);
    // status = clEnqueueReleaseGLObjects(CmdQueue, 1, &dCobj, 0,
    //                                    NULL, NULL);
    // PrintCLError(status, "clEnqueueReleaseGLObjects (2): ");
    // status = clEnqueueReleaseGLObjects(CmdQueue, 1, &dPobj, 0,
    //                                    NULL, NULL);
    // PrintCLError(status, "clEnqueueReleaseGLObjects (2): ");

    glutSetWindow(MainWindow);
    glutPostRedisplay();
}

// glui buttons callback.
void Buttons(int id) {
    cl_int status;
    switch (id) {
        case GO:
            GLUI_Master.set_glutIdleFunc(Animate);
            break;

        case PAUSE:
            Paused = !Paused;
            if (Paused)
                GLUI_Master.set_glutIdleFunc(NULL);
            else
                GLUI_Master.set_glutIdleFunc(Animate);
            break;

        case RESET:
            Reset();
            ResetParticles();
            status = clEnqueueWriteBuffer(
                CmdQueue, dVel, CL_FALSE, 0,
                4 * sizeof(float) * NUM_PARTICLES, hVel, 0, NULL,
                NULL);
            PrintCLError(status, "clEneueueWriteBuffer: ");
            GLUI_Master.set_glutIdleFunc(NULL);
            Glui->sync_live();
            glutSetWindow(MainWindow);
            glutPostRedisplay();
            break;

        case QUIT:
            Quit();
            break;

        default:
            fprintf(stderr,
                    "Don't know what to do with Button ID %d\n",
                    id);
    }
}

// Draw the complete scene.
void DrawScene() {
    glutSetWindow(MainWindow);
    glDrawBuffer(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_FLAT);
    GLsizei vx = glutGet(GLUT_WINDOW_WIDTH);
    GLsizei vy = glutGet(GLUT_WINDOW_HEIGHT);
    GLsizei v = vx < vy ? vx : vy;  // minimum dimension
    GLint xl = (vx - v) / 2;
    GLint yb = (vy - v) / 2;
    glViewport(xl, yb, v, v);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (WhichProjection == ORTHO)
        glOrtho(-300., 300., -300., 300., 0.1, 2000.);
    else
        gluPerspective(50., 1., 0.1, 2000.);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0., -100., 800., 0., -100., 0., 0., 1., 0.);
    glTranslatef((GLfloat)TransXYZ[0], (GLfloat)TransXYZ[1],
                 -(GLfloat)TransXYZ[2]);
    glRotatef((GLfloat)Yrot, 0., 1., 0.);
    glRotatef((GLfloat)Xrot, 1., 0., 0.);
    glMultMatrixf((const GLfloat *)RotMatrix);
    glScalef((GLfloat)Scale, (GLfloat)Scale, (GLfloat)Scale);
    float scale2 =
        1.f + Scale2;  // because glui translation starts at 0.
    if (scale2 < MINSCALE)
        scale2 = MINSCALE;
    glScalef((GLfloat)scale2, (GLfloat)scale2, (GLfloat)scale2);

    glDisable(GL_FOG);

    if (AxesOn != GLUIFALSE)
        glCallList(AxesList);

    // ****************************************
    // Here is where you draw the current state of the particles:
    // ****************************************

    glBindBuffer(GL_ARRAY_BUFFER, hPobj);
    glVertexPointer(4, GL_FLOAT, 0, (void *)0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, hCobj);
    glColorPointer(4, GL_FLOAT, 0, (void *)0);
    glEnableClientState(GL_COLOR_ARRAY);

    glPointSize(3.);
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
    glPointSize(1.);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ****************************************
    // Here is where you draw your spheres.  You define them,
    // however, in the InitLists( ) function.
    // ****************************************

    glCallList(SphereList);

    if (ShowPerformance) {
        char str[128];
        sprintf(
            str, "%6.1f GigaParticles/Sec",
            (float)NUM_PARTICLES / ElapsedTime / 1000000000.);
        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0., 100., 0., 100.);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glColor3f(1., 1., 1.);
        DoRasterString(5., 95., 0., str);
    }

    glutSwapBuffers();
    glFlush();
}

// Use glut to display a string of characters using a raster
// font.
void DoRasterString(float x, float y, float z, char *s) {
    char c;  // one character to print

    glRasterPos3f((GLfloat)x, (GLfloat)y, (GLfloat)z);
    for (; (c = *s) != '\0'; s++) {
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, c);
    }
}

// Use glut to display a string of characters using a stroke
// font.
void DoStrokeString(
    float x, float y, float z, float ht, char *s) {
    char c;  // one character to print

    glPushMatrix();
    glTranslatef((GLfloat)x, (GLfloat)y, (GLfloat)z);
    float sf = ht / (119.05f + 33.33f);
    glScalef((GLfloat)sf, (GLfloat)sf, (GLfloat)sf);
    for (; (c = *s) != '\0'; s++) {
        glutStrokeCharacter(GLUT_STROKE_ROMAN, c);
    }
    glPopMatrix();
}

// Initialize the OpenCL stuff.
bool InitCL() {

    // Show us what we've got here.
    PrintOpenclInfo();

    // Pick which OpenCL device to use.
    // This fills the globals Platform and Device.
    SelectOpenclDevice();

    // See if we can even open the OpenCL Kernel Program
    // (no point going on if we can't).
    FILE *fp = fopen(CL_FILE_NAME, "r");
    if (fp == NULL) {
        fprintf(stderr, "Cannot open OpenCL source file '%s'\n",
                CL_FILE_NAME);
        return false;
    }

    // Allocate the host memory buffers.
    cl_int status;  // returned status from opencl calls
                    // test against CL_SUCCESS

    // Since this is an OpenGL interoperability program,
    // check if the OpenGL sharing extension is supported.
    // We need the Device in order to ask, so can't do it any
    // sooner than here.

    if (IsCLExtensionSupported("cl_khr_gl_sharing")) {
        fprintf(stderr, "cl_khr_gl_sharing is supported.\n");
    } else {
        fprintf(
            stderr,
            "cl_khr_gl_sharing is not supported -- sorry.\n");
        return false;
    }

    unsigned int glDeviceCount = 2;
    int glDevices[2];

    // Print all devices that are spanned by the current GL context.
    cudaGLGetDevices(&glDeviceCount, glDevices, glDeviceCount, cudaGLDeviceListAll);
    CUDA_CHECK_ERROR();
    printf("OpenGL is using CUDA device(s): ");
    for (unsigned int i = 0; i < glDeviceCount; ++i) {
        printf("%s%d", i == 0 ? "" : ", ", glDevices[i]);
    }

    // Create the velocity array and the OpenGL vertex array
    // buffer and color array buffer.
    delete[] hVel;
    hVel = new struct xyzw[NUM_PARTICLES];

    glGenBuffers(1, &hPobj);
    glBindBuffer(GL_ARRAY_BUFFER, hPobj);
    glBufferData(GL_ARRAY_BUFFER,
                 4 * NUM_PARTICLES * sizeof(float), NULL,
                 GL_STATIC_DRAW);

    glGenBuffers(1, &hCobj);
    glBindBuffer(GL_ARRAY_BUFFER, hCobj);
    glBufferData(GL_ARRAY_BUFFER,
                 4 * NUM_PARTICLES * sizeof(float), NULL,
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);  // unbind the buffer

    // Fill those arrays and buffers.
    ResetParticles();



    // Create the OpenCL version of the OpenGL buffers.
    cudaGraphicsGLRegisterBuffer(&dPobj, hPobj, cudaGraphicsRegisterFlagsNone);
    CUDA_CHECK_ERROR();

    // dPobj = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE,
    //                              hPobj, &status);
    // PrintCLError(status, "clCreateFromGLBuffer (1)");

    cudaGraphicsGLRegisterBuffer(&dCobj, hCobj, cudaGraphicsRegisterFlagsNone);
    CUDA_CHECK_ERROR();

    // dCobj = clCreateFromGLBuffer(Context, CL_MEM_READ_WRITE,
    //                              hCobj, &status);
    // PrintCLError(status, "clCreateFromGLBuffer (2)");

    // Create the OpenCL version of the velocity array.
    
    CudaRig::InitAndCopy(dVel, hVel, 4 * sizeof(float) * NUM_PARTICLES);
    CUDA_CHECK_ERROR();
    // dVel = clCreateBuffer(Context, CL_MEM_READ_WRITE,
    //                       4 * sizeof(float) * NUM_PARTICLES,
    //                       NULL, &status);
    // PrintCLError(status, "clCreateBuffer: ");

    // // Enqueue the command to write the data from the host
    // // buffers to the Device buffers.
    // status = clEnqueueWriteBuffer(
    //     CmdQueue, dVel, CL_FALSE, 0,
    //     4 * sizeof(float) * NUM_PARTICLES, hVel, 0, NULL, NULL);
    // PrintCLError(status, "clEneueueWriteBuffer: ");
}

// Initialize the glui window.
void InitGlui() {
    glutInitWindowPosition(INIT_WINDOW_SIZE + 50, 0);
    Glui = GLUI_Master.create_glui((char *)GLUITITLE);
    Glui->add_statictext((char *)GLUITITLE);
    Glui->add_separator();
    Glui->add_checkbox("Axes", &AxesOn);
    Glui->add_checkbox("Perspective", &WhichProjection);
    Glui->add_checkbox("Show Performance", &ShowPerformance);

    GLUI_Panel *panel = Glui->add_panel("Object Transformation");

    GLUI_Rotation *rot = Glui->add_rotation_to_panel(
        panel, "Rotation", (float *)RotMatrix);
    rot->set_spin(1.0);

    Glui->add_column_to_panel(panel, GLUIFALSE);
    GLUI_Translation *scale = Glui->add_translation_to_panel(
        panel, "Scale", GLUI_TRANSLATION_Y, &Scale2);
    scale->set_speed(0.01f);

    Glui->add_column_to_panel(panel, GLUIFALSE);
    GLUI_Translation *trans = Glui->add_translation_to_panel(
        panel, "Trans XY", GLUI_TRANSLATION_XY, &TransXYZ[0]);
    trans->set_speed(1.1f);

    Glui->add_column_to_panel(panel, GLUIFALSE);
    trans = Glui->add_translation_to_panel(
        panel, "Trans Z", GLUI_TRANSLATION_Z, &TransXYZ[2]);
    trans->set_speed(1.1f);

    panel = Glui->add_panel("", GLUIFALSE);
    Glui->add_button_to_panel(panel, "Go !", GO,
                              (GLUI_Update_CB)Buttons);
    Glui->add_column_to_panel(panel, GLUIFALSE);
    Glui->add_button_to_panel(panel, "Pause", PAUSE,
                              (GLUI_Update_CB)Buttons);
    Glui->add_column_to_panel(panel, GLUIFALSE);
    Glui->add_button_to_panel(panel, "Reset", RESET,
                              (GLUI_Update_CB)Buttons);
    Glui->add_column_to_panel(panel, GLUIFALSE);
    Glui->add_button_to_panel(panel, "Quit", QUIT,
                              (GLUI_Update_CB)Buttons);

    Glui->set_main_gfx_window(MainWindow);
    GLUI_Master.set_glutIdleFunc(NULL);
}

// Initialize the glut and OpenGL libraries and also setup 
// display lists and callback functions.
void InitGraphics() {
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(INIT_WINDOW_SIZE, INIT_WINDOW_SIZE);

    MainWindow = glutCreateWindow(WINDOWTITLE);
    glutSetWindowTitle(WINDOWTITLE);
    glClearColor(BACKCOLOR[0], BACKCOLOR[1], BACKCOLOR[2],
                 BACKCOLOR[3]);

    // Setup the callback routines.
    glutSetWindow(MainWindow);
    glutDisplayFunc(DrawScene);
    glutReshapeFunc(Resize);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(MouseButton);
    glutMotionFunc(MouseMotion);
    glutVisibilityFunc(Visibility);

#ifdef WIN32
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "glewInit Error\n");
    }
#endif
}

// Initialize the display lists that will not change.
void InitLists() {
    SphereList = glGenLists(1);
    glNewList(SphereList, GL_COMPILE);

    // ****************************************
    // Here is where the first sphere is drawn.
    // It is centered at (-100,-800.,0.) with a radius of 600.
    // This had better match the definition of the first sphere
    // 	in the .cl file
    // ****************************************

    glColor3f(.9f, .9f, 0.);  // 0. <= r,g,b <= 1.
    glPushMatrix();
    glTranslatef(-100.f, -800.f, 0.f);
    glutWireSphere(600.f, 100, 100);
    glPopMatrix();

    // ****************************************
    // Here is where the second sphere is drawn.
    // This had better match the definition of the second sphere
    // 	in the .cl file
    // ****************************************

    // do this for yourself...

    glEndList();

    AxesList = glGenLists(1);
    glNewList(AxesList, GL_COMPILE);
    glColor3fv(AXES_COLOR);
    glLineWidth(AXES_WIDTH);
    Axes(150.);
    glLineWidth(1.);
    glEndList();
}

// The keyboard callback.
void Keyboard(unsigned char c, int x, int y) {
    switch (c) {
        case 'o':
        case 'O':
            WhichProjection = ORTHO;
            break;

        case 'p':
        case 'P':
            WhichProjection = PERSP;
            break;

        case 'q':
        case 'Q':
        case ESCAPE:
            Buttons(QUIT);  // will not return here
            break;          // happy compiler

        default:
            fprintf(stderr,
                    "Don't know what to do with keyboard hit: "
                    "'%c' (0x%0x)\n",
                    c, c);
    }
    Glui->sync_live();
    glutSetWindow(MainWindow);
    glutPostRedisplay();
}

// Called when the mouse button transitions down or up.
void MouseButton(int button, int state, int x, int y) {
    int b;  // LEFT, MIDDLE, or RIGHT

    switch (button) {
        case GLUT_LEFT_BUTTON:
            b = LEFT;
            break;

        case GLUT_MIDDLE_BUTTON:
            b = MIDDLE;
            break;

        case GLUT_RIGHT_BUTTON:
            b = RIGHT;
            break;

        default:
            b = 0;
            fprintf(stderr, "Unknown mouse button: %d\n",
                    button);
    }

    // Button down sets the bit, up clears the bit.
    if (state == GLUT_DOWN) {
        Xmouse = x;
        Ymouse = y;
        ActiveButton |= b;  // set the proper bit
    } else {
        ActiveButton &= ~b;  // clear the proper bit
    }
}

// Called when the mouse moves while a button is down.
void MouseMotion(int x, int y) {
    int dx = x - Xmouse;  // change in mouse coords
    int dy = y - Ymouse;

    if (ActiveButton & LEFT) {
        Xrot += (ANGFACT * dy);
        Yrot += (ANGFACT * dx);
    }

    if (ActiveButton & MIDDLE) {
        Scale += SCLFACT * (float)(dx - dy);

        // Keep object from turning inside-out or disappearing.
        if (Scale < MINSCALE)
            Scale = MINSCALE;
    }

    Xmouse = x;  // new current position
    Ymouse = y;

    glutSetWindow(MainWindow);
    glutPostRedisplay();
}

// Reset the transformations and the colors. This only sets 
// the global variables -- the glut main loop is responsible 
// for redrawing the scene.
void Reset() {
    ActiveButton = 0;
    AxesOn = GLUIFALSE;
    Paused = GLUIFALSE;
    Scale = 1.0;
    Scale2 = 0.0;  // because add 1. to it in Display( )
    ShowPerformance = GLUIFALSE;
    WhichProjection = PERSP;
    Xrot = Yrot = 0.;
    TransXYZ[0] = TransXYZ[1] = TransXYZ[2] = 0.;

    RotMatrix[0][1] = RotMatrix[0][2] = RotMatrix[0][3] = 0.;
    RotMatrix[1][0] = RotMatrix[1][2] = RotMatrix[1][3] = 0.;
    RotMatrix[2][0] = RotMatrix[2][1] = RotMatrix[2][3] = 0.;
    RotMatrix[3][0] = RotMatrix[3][1] = RotMatrix[3][3] = 0.;
    RotMatrix[0][0] = RotMatrix[1][1] = RotMatrix[2][2] =
        RotMatrix[3][3] = 1.;
}

void ResetParticles() {
    glBindBuffer(GL_ARRAY_BUFFER, hPobj);
    struct xyzw *points = (struct xyzw *)glMapBuffer(
        GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        points[i].x = Ranf(XMIN, XMAX);
        points[i].y = Ranf(YMIN, YMAX);
        points[i].z = Ranf(ZMIN, ZMAX);
        points[i].w = 1.;
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);

    glBindBuffer(GL_ARRAY_BUFFER, hCobj);
    struct rgba *colors = (struct rgba *)glMapBuffer(
        GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        colors[i].r = Ranf(.3f, 1.);
        colors[i].g = Ranf(.3f, 1.);
        colors[i].b = Ranf(.3f, 1.);
        colors[i].a = 1.;
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);

    for (int i = 0; i < NUM_PARTICLES; i++) {
        hVel[i].x = Ranf(VMIN, VMAX);
        hVel[i].y = Ranf(0., VMAX);
        hVel[i].z = Ranf(VMIN, VMAX);
    }
}

void Resize(int width, int height) {
    glutSetWindow(MainWindow);
    glutPostRedisplay();
}

void Visibility(int state) {
    if (state == GLUT_VISIBLE) {
        glutSetWindow(MainWindow);
        glutPostRedisplay();
    } else {
        // could optimize by keeping track of the fact
        // that the window is not visible and avoid
        // animating or redrawing it ...
    }
}

// the stroke characters 'X' 'Y' 'Z' :

static float xx[] = {0., 1., 0., 1.};

static float xy[] = {-.5, .5, .5, -.5};

static int xorder[] = {1, 2, -3, 4};

static float yx[] = {0., 0., -.5, .5};

static float yy[] = {0., .6f, 1., 1.};

static int yorder[] = {1, 2, 3, -2, 4};

static float zx[] = {1., 0., 1., 0., .25, .75};

static float zy[] = {.5, .5, -.5, -.5, 0., 0.};

static int zorder[] = {1, 2, 3, 4, -5, 6};

// fraction of the length to use as height of the characters:

#define LENFRAC 0.10

// fraction of length to use as start location of the characters:

#define BASEFRAC 1.10

// Draw a set of 3D axes. (length is the axis length in 
// world coordinates)
void Axes(float length) {
    glBegin(GL_LINE_STRIP);
    glVertex3f(length, 0., 0.);
    glVertex3f(0., 0., 0.);
    glVertex3f(0., length, 0.);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(0., 0., 0.);
    glVertex3f(0., 0., length);
    glEnd();

    float fact = (float)LENFRAC * length;
    float base = (float)BASEFRAC * length;

    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < 4; i++) {
        int j = xorder[i];
        if (j < 0) {
            glEnd();
            glBegin(GL_LINE_STRIP);
            j = -j;
        }
        j--;
        glVertex3f(base + fact * xx[j], fact * xy[j], 0.0);
    }
    glEnd();

    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < 5; i++) {
        int j = yorder[i];
        if (j < 0) {
            glEnd();
            glBegin(GL_LINE_STRIP);
            j = -j;
        }
        j--;
        glVertex3f(fact * yx[j], base + fact * yy[j], 0.0);
    }
    glEnd();

    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < 6; i++) {
        int j = zorder[i];
        if (j < 0) {
            glEnd();
            glBegin(GL_LINE_STRIP);
            j = -j;
        }
        j--;
        glVertex3f(0.0, fact * zy[j], base + fact * zx[j]);
    }
    glEnd();
}

// Exit gracefully.
void Quit() {
    Glui->close();
    glutSetWindow(MainWindow);
    glFinish();
    glutDestroyWindow(MainWindow);

    // Clean everything up.
    clReleaseKernel(Kernel);
    clReleaseProgram(Program);
    clReleaseCommandQueue(CmdQueue);
    clReleaseMemObject(dPobj);
    clReleaseMemObject(dCobj);

    exit(0);
}

#define TOP 2147483647.  // 2^31 - 1

float Ranf(float low, float high) {
    long random();  // returns integer 0 - TOP

    float r = (float)rand();
    return (low + r * (high - low) / (float)RAND_MAX);
}

bool IsCLExtensionSupported(const char *extension) {
    // See if the extension is bogus.
    if (extension == NULL || extension[0] == '\0')
        return false;

    char *where = (char *)strchr(extension, ' ');
    if (where != NULL)
        return false;

    // Get the full list of extensions.
    size_t extensionSize;
    clGetDeviceInfo(Device, CL_DEVICE_EXTENSIONS, 0, NULL,
                    &extensionSize);
    char *extensions = new char[extensionSize];
    clGetDeviceInfo(Device, CL_DEVICE_EXTENSIONS, extensionSize,
                    extensions, NULL);

    for (char *start = extensions;;) {
        where = (char *)strstr((const char *)start, extension);
        if (where == 0) {
            delete[] extensions;
            return false;
        }

        char *terminator =
            where + strlen(extension);  // points to what should
                                        // be the separator

        if (*terminator == ' ' || *terminator == '\0' ||
            *terminator == '\r' || *terminator == '\n') {
            delete[] extensions;
            return true;
        }
        start = terminator;
    }

    delete[] extensions;
    return false;
}

struct errorcode {
    cl_int statusCode;
    const char *meaning;
} ErrorCodes[] = {
    {CL_SUCCESS, ""},
    {CL_DEVICE_NOT_FOUND, "Device Not Found"},
    {CL_DEVICE_NOT_AVAILABLE, "Device Not Available"},
    {CL_COMPILER_NOT_AVAILABLE, "Compiler Not Available"},
    {CL_MEM_OBJECT_ALLOCATION_FAILURE,
     "Memory Object Allocation Failure"},
    {CL_OUT_OF_RESOURCES, "Out of resources"},
    {CL_OUT_OF_HOST_MEMORY, "Out of Host Memory"},
    {CL_PROFILING_INFO_NOT_AVAILABLE,
     "Profiling Information Not Available"},
    {CL_MEM_COPY_OVERLAP, "Memory Copy Overlap"},
    {CL_IMAGE_FORMAT_MISMATCH, "Image Format Mismatch"},
    {CL_IMAGE_FORMAT_NOT_SUPPORTED,
     "Image Format Not Supported"},
    {CL_BUILD_PROGRAM_FAILURE, "Build Program Failure"},
    {CL_MAP_FAILURE, "Map Failure"},
    {CL_INVALID_VALUE, "Invalid Value"},
    {CL_INVALID_DEVICE_TYPE, "Invalid Device Type"},
    {CL_INVALID_PLATFORM, "Invalid Platform"},
    {CL_INVALID_DEVICE, "Invalid Device"},
    {CL_INVALID_CONTEXT, "Invalid Context"},
    {CL_INVALID_QUEUE_PROPERTIES, "Invalid Queue Properties"},
    {CL_INVALID_COMMAND_QUEUE, "Invalid Command Queue"},
    {CL_INVALID_HOST_PTR, "Invalid Host Pointer"},
    {CL_INVALID_MEM_OBJECT, "Invalid Memory Object"},
    {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
     "Invalid Image Format Descriptor"},
    {CL_INVALID_IMAGE_SIZE, "Invalid Image Size"},
    {CL_INVALID_SAMPLER, "Invalid Sampler"},
    {CL_INVALID_BINARY, "Invalid Binary"},
    {CL_INVALID_BUILD_OPTIONS, "Invalid Build Options"},
    {CL_INVALID_PROGRAM, "Invalid Program"},
    {CL_INVALID_PROGRAM_EXECUTABLE,
     "Invalid Program Executable"},
    {CL_INVALID_KERNEL_NAME, "Invalid Kernel Name"},
    {CL_INVALID_KERNEL_DEFINITION, "Invalid Kernel Definition"},
    {CL_INVALID_KERNEL, "Invalid Kernel"},
    {CL_INVALID_ARG_INDEX, "Invalid Argument Index"},
    {CL_INVALID_ARG_VALUE, "Invalid Argument Value"},
    {CL_INVALID_ARG_SIZE, "Invalid Argument Size"},
    {CL_INVALID_KERNEL_ARGS, "Invalid Kernel Arguments"},
    {CL_INVALID_WORK_DIMENSION, "Invalid Work Dimension"},
    {CL_INVALID_WORK_GROUP_SIZE, "Invalid Work Group Size"},
    {CL_INVALID_WORK_ITEM_SIZE, "Invalid Work Item Size"},
    {CL_INVALID_GLOBAL_OFFSET, "Invalid Global Offset"},
    {CL_INVALID_EVENT_WAIT_LIST, "Invalid Event Wait List"},
    {CL_INVALID_EVENT, "Invalid Event"},
    {CL_INVALID_OPERATION, "Invalid Operation"},
    {CL_INVALID_GL_OBJECT, "Invalid GL Object"},
    {CL_INVALID_BUFFER_SIZE, "Invalid Buffer Size"},
    {CL_INVALID_MIP_LEVEL, "Invalid MIP Level"},
    {CL_INVALID_GLOBAL_WORK_SIZE, "Invalid Global Work Size"},
    {CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR, "CL and GL not on the same device (only when using a GPU)."}
};

void PrintCLError(cl_int errorCode, const char *prefix, FILE *fp) {
    if (errorCode == CL_SUCCESS)
        return;

    const int numErrorCodes =
        sizeof(ErrorCodes) / sizeof(struct errorcode);
    const char *meaning = "";
    for (int i = 0; i < numErrorCodes; i++) {
        if (errorCode == ErrorCodes[i].statusCode) {
            meaning = ErrorCodes[i].meaning;
            break;
        }
    }

    fprintf(fp, "%s %s\n", prefix, meaning);
}

void PrintOpenclInfo() {
    cl_int status;  // returned status from opencl calls
                    // test against CL_SUCCESS
    fprintf(stderr, "PrintInfo:\n");

    // Find out how many platforms are attached here and get
    // their ids.
    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clGetPlatformIDs failed (1)\n");

    fprintf(stderr, "Number of Platforms = %d\n", numPlatforms);

    cl_platform_id *platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clGetPlatformIDs failed (2)\n");

    for (int p = 0; p < (int)numPlatforms; p++) {
        fprintf(stderr, "Platform #%d:\n", p);
        size_t size;
        char *str;

        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 0,
                          NULL, &size);
        str = new char[size];
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, size,
                          str, NULL);
        fprintf(stderr, "\tName    = '%s'\n", str);
        delete[] str;

        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 0,
                          NULL, &size);
        str = new char[size];
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, size,
                          str, NULL);
        fprintf(stderr, "\tVendor  = '%s'\n", str);
        delete[] str;

        clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, 0,
                          NULL, &size);
        str = new char[size];
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION,
                          size, str, NULL);
        fprintf(stderr, "\tVersion = '%s'\n", str);
        delete[] str;

        clGetPlatformInfo(platforms[p], CL_PLATFORM_PROFILE, 0,
                          NULL, &size);
        str = new char[size];
        clGetPlatformInfo(platforms[p], CL_PLATFORM_PROFILE,
                          size, str, NULL);
        fprintf(stderr, "\tProfile = '%s'\n", str);
        delete[] str;

        // find out how many devices are attached to each
        // platform and get their ids:

        cl_uint numDevices;

        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                0, NULL, &numDevices);
        if (status != CL_SUCCESS)
            fprintf(stderr, "clGetDeviceIDs failed (2)\n");

        fprintf(stderr, "\tNumber of Devices = %d\n",
                numDevices);

        cl_device_id *devices = new cl_device_id[numDevices];
        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                numDevices, devices, NULL);
        if (status != CL_SUCCESS)
            fprintf(stderr, "clGetDeviceIDs failed (2)\n");

        for (int d = 0; d < (int)numDevices; d++) {
            fprintf(stderr, "\tDevice #%d:\n", d);
            size_t size;
            cl_device_type type;
            cl_uint ui;
            size_t sizes[3] = {0, 0, 0};

            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE,
                            sizeof(type), &type, NULL);
            fprintf(stderr, "\t\tType = 0x%04x = ",
                    (unsigned int)type);
            switch (type) {
                case CL_DEVICE_TYPE_CPU:
                    fprintf(stderr, "CL_DEVICE_TYPE_CPU\n");
                    break;
                case CL_DEVICE_TYPE_GPU:
                    fprintf(stderr, "CL_DEVICE_TYPE_GPU\n");
                    break;
                case CL_DEVICE_TYPE_ACCELERATOR:
                    fprintf(stderr,
                            "CL_DEVICE_TYPE_ACCELERATOR\n");
                    break;
                default:
                    fprintf(stderr, "Other...\n");
                    break;
            }

            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID,
                            sizeof(ui), &ui, NULL);
            fprintf(stderr, "\t\tDevice Vendor ID = 0x%04x ",
                    ui);
            switch (ui) {
                case ID_AMD:
                    fprintf(stderr, "(AMD)\n");
                    break;
                case ID_INTEL:
                    fprintf(stderr, "(Intel)\n");
                    break;
                case ID_NVIDIA:
                    fprintf(stderr, "(NVIDIA)\n");
                    break;
                default:
                    fprintf(stderr, "(?)\n");
            }

            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(ui), &ui, NULL);
            fprintf(stderr,
                    "\t\tDevice Maximum Compute Units = %d\n",
                    ui);

            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(ui), &ui, NULL);
            fprintf(
                stderr,
                "\t\tDevice Maximum Work Item Dimensions = %d\n",
                ui);

            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            sizeof(sizes), sizes, NULL);
            fprintf(stderr,
                    "\t\tDevice Maximum Work Item Sizes = %zu x "
                    "%zu x %zu\n",
                    sizes[0], sizes[1], sizes[2]);

            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof(size), &size, NULL);
            fprintf(stderr,
                    "\t\tDevice Maximum Work Group Size = %zu\n",
                    size);

            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(ui), &ui, NULL);
            fprintf(
                stderr,
                "\t\tDevice Maximum Clock Frequency = %d MHz\n",
                ui);

            size_t extensionSize;
            clGetDeviceInfo(devices[d], CL_DEVICE_EXTENSIONS, 0,
                            NULL, &extensionSize);
            char *extensions = new char[extensionSize];
            clGetDeviceInfo(devices[d], CL_DEVICE_EXTENSIONS,
                            extensionSize, extensions, NULL);
            fprintf(stderr, "\nDevice #%d's Extensions:\n", d);
            for (int e = 0; e < (int)strlen(extensions); e++) {
                if (extensions[e] == ' ')
                    extensions[e] = '\n';
            }
            fprintf(stderr, "%s\n", extensions);
            delete[] extensions;
        }
        delete[] devices;
    }
    delete[] platforms;
    fprintf(stderr, "\n\n");
}

void SelectOpenclDevice() {
    // select which opencl device to use:
    // priority order:
    //	1. a gpu
    //	2. an nvidia or amd gpu

    int iplatform = -1;
    int idevice = -1;
    cl_device_type deviceType;
    cl_uint deviceVendor;
    cl_int status;  // returned status from opencl calls
                    // test against CL_SUCCESS

    fprintf(stderr,
            "\nSelecting the OpenCL Platform and Device:\n");

    // find out how many platforms are attached here and get
    // their ids:

    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clGetPlatformIDs failed (1)\n");

    cl_platform_id *platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS)
        fprintf(stderr, "clGetPlatformIDs failed (2)\n");

    for (int p = 0; p < (int)numPlatforms; p++) {
        // find out how many devices are attached to each
        // platform and get their ids:
        cl_uint numDevices;

        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                0, NULL, &numDevices);
        if (status != CL_SUCCESS)
            fprintf(stderr, "clGetDeviceIDs failed (2)\n");

        cl_device_id *devices = new cl_device_id[numDevices];
        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                numDevices, devices, NULL);
        if (status != CL_SUCCESS)
            fprintf(stderr, "clGetDeviceIDs failed (2)\n");

        for (int d = 0; d < (int)numDevices; d++) {
            cl_device_type type;
            cl_uint vendor;
            size_t sizes[3] = {0, 0, 0};

            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE,
                            sizeof(type), &type, NULL);

            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID,
                            sizeof(vendor), &vendor, NULL);

            // select:

            if (iplatform < 0)  // not yet holding anything --
                                // we'll accept anything
            {
                iplatform = p;
                idevice = d;
                Platform = platforms[iplatform];
                Device = devices[idevice];
                deviceType = type;
                deviceVendor = vendor;
            } else  // holding something already -- can we do
                    // better?
            {
                if (deviceType ==
                    CL_DEVICE_TYPE_CPU)  // holding a cpu already
                                         // -- switch to a gpu if
                                         // possible
                {
                    if (type ==
                        CL_DEVICE_TYPE_GPU)  // found a gpu
                    {  // switch to the gpu we just found
                        iplatform = p;
                        idevice = d;
                        Platform = platforms[iplatform];
                        Device = devices[idevice];
                        deviceType = type;
                        deviceVendor = vendor;
                    }
                } else  // holding a gpu -- is a better gpu
                        // available?
                {
                    if (deviceVendor == ID_INTEL)  // currently
                                                   // holding an
                                                   // intel gpu
                    {  // we are assuming we just found a bigger,
                       // badder nvidia or amd gpu
                        iplatform = p;
                        idevice = d;
                        Platform = platforms[iplatform];
                        Device = devices[idevice];
                        deviceType = type;
                        deviceVendor = vendor;
                    }
                }
            }
        }
    }

    if (iplatform < 0)
        fprintf(stderr, "Found no OpenCL devices!\n");
    else
        fprintf(stderr,
                "I have selected Platform #%d, Device #%d\n",
                iplatform, idevice);
}
