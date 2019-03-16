#include <cstdio>
#include "Interpreter.h"
#include "CudaModule.h"
#include "IntItem.h"
#include "Dim3Item.h"


// =============================================================================
// Kernels
__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}


__global__ void checkIndex() {
    printf("blockIdx:(%d, %d, %d) threadIdx:(%d, %d, %d) blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
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


// ( dim3 -- coord )
class ToCoordWord : public Word
{
public:
    ToCoordWord(string name, string coord) : Word(name), coord(coord) {};

    virtual void Execute(Interpreter *interp) {
        dim3 d = AsDim3(interp->StackPop());

        int res = -1;
        if      (coord == "x")   res = d.x;
        else if (coord == "y")   res = d.y;
        else if (coord == "z")   res = d.z;
        else                     throw string("Unknown coord: ") + coord;

        interp->StackPush(shared_ptr<IntItem>(new IntItem(res)));
    }

protected:
    string coord;
};


// ( grid block -- )
class CheckIndexWord : public Word
{
public:
    CheckIndexWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        dim3 block = AsDim3(interp->StackPop());
        dim3 grid = AsDim3(interp->StackPop()); 

        checkIndex<<<grid, block>>>();
        cudaDeviceReset();
    }
};

// =============================================================================
// CudaModule

CudaModule::CudaModule() : Module("cuda")
{
    AddWord(shared_ptr<Word>(new HelloWord("HELLO")));
    AddWord(shared_ptr<Word>(new Dim3Word("DIM3")));
    AddWord(shared_ptr<Word>(new ToCoordWord(">x", "x")));
    AddWord(shared_ptr<Word>(new ToCoordWord(">y", "y")));
    AddWord(shared_ptr<Word>(new ToCoordWord(">z", "z")));
    AddWord(shared_ptr<Word>(new CheckIndexWord("GPU-CHECK-INDEX")));
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

