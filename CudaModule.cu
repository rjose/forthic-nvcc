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
        int threads_per_block = AsInt(interp->StackPop());
        int num_blocks = AsInt(interp->StackPop()); 

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

