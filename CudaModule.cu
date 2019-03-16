#include <cstdio>
#include "Interpreter.h"
#include "CudaModule.h"
#include "Dim3Item.h"


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

// ( x y z -- dim3 )
class Dim3Word : public Word
{
public:
    Dim3Word(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int z = AsInt(interp->StackPop());
        int y = AsInt(interp->StackPop());
        int x = AsInt(interp->StackPop());
        dim3 res(x, y, z);

        interp->StackPush(shared_ptr<Dim3Item>(new Dim3Item(res)));
    }
};


// =============================================================================
// CudaModule

CudaModule::CudaModule() : Module("cuda")
{
    AddWord(shared_ptr<Word>(new HelloWord("HELLO")));
    AddWord(shared_ptr<Word>(new Dim3Word("DIM3")));
}

// =============================================================================
// StackItem Converters


dim3 AsDim3(shared_ptr<StackItem> item)
{
    if (auto i = dynamic_cast<IGetDim3*>(item.get()))
    {
        return i->GetDim3();
    }
    else
    {
        throw "Item does not implement IGetInt";
    }
}

