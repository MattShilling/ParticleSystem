#ifndef CL_RIG_
#define CL_RIG_

#include "CL/cl.h"

#include <string>

#define ASSERT(X)                                  \
    do {                                           \
        if ((X) == false) {                        \
            fprintf(stderr,                        \
                    "[%s:%d] -> (%s) is false.\n", \
                    __FILE__,                      \
                    __LINE__,                      \
                    #X);                           \
            return false;                          \
        }                                          \
    } while (false);

#define ASSERT_MSG(X, M)                        \
    do {                                        \
        if ((X) == false) {                     \
            fprintf(stderr,                     \
                    "[%s:%d] -> (%s) is false." \
                    "\n\t\t\\-> %s\n",          \
                    __FILE__,                   \
                    __LINE__,                   \
                    #X,                         \
                    M);                         \
            return false;                       \
        }                                       \
    } while (false);

#define CHECK_CL(X, ...)               \
    if (X != CL_SUCCESS) {             \
        fprintf(stderr,                \
                "[%s:%d][%d] -> %s\n", \
                __FILE__,              \
                __LINE__,              \
                (int)(X),              \
                __VA_ARGS__);          \
        return false;                  \
    }

class ClRig {
  public:
    ClRig() : initialized_(false) {}

    ~ClRig() {
        clReleaseKernel(kernel_);
        clReleaseProgram(program_);
        clReleaseCommandQueue(cmd_queue_);
    }

    bool Init();

    bool CreateReadBuffer(cl_mem *mem, size_t data_size);
    bool CreateWriteBuffer(cl_mem *mem, size_t data_size);

    bool EnqueueWriteBuffer(cl_mem d,
                            size_t data_size,
                            const void *ptr);

    // Wait until all queued tasks have taken
    // place:
    bool Wait();

    // Step 7: Read the kernel code from a file.
    bool AddProgramFile(const char *);
    bool CreateProgram();
    bool CreateProgram(const std::string &source);

    bool BuildProgram(const std::string &options,
                      const std::string &k_name);

    bool SetKernelArg(cl_uint arg_index,
                      size_t arg_size,
                      const void *arg);

    bool GetCommandQueue(cl_command_queue *cmd_queue);

    bool GetKernel(cl_kernel *kernel);

  private:
    const char *program_path_;
    cl_platform_id platform_;
    cl_device_id device_;
    FILE *fp_;
    cl_int status_;
    cl_context context_;
    cl_command_queue cmd_queue_;
    bool initialized_;
    cl_program program_;
    cl_kernel kernel_;
};

#endif  // CL_RIG_