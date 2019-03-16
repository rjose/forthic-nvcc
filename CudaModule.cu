#include <cuda_runtime.h>
#include <cstdio>
#include "Interpreter.h"
#include "CudaModule.h"


// =============================================================================
// Kernels
__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}


// =============================================================================
// Words

// ( num_blocks thread_per_block -- )
class HelloWord : public Word
{
public:
    HelloWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int threads_per_block = ForthicGetInt(interp->StackPop().get());
        int num_blocks = ForthicGetInt(interp->StackPop().get()); 
        helloFromGPU<<<num_blocks, threads_per_block>>>();
        cudaDeviceReset();
    }
};


// =============================================================================
// CudaModule

CudaModule::CudaModule() : Module("cuda")
{
    AddWord(shared_ptr<Word>(new HelloWord("HELLO")));
}

