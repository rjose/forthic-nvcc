#include <cstdio>
#include <sstream>

#include "Interpreter.h"
#include "CudaModule.h"
#include "IntItem.h"
#include "Dim3Item.h"
#include "AddressItem.h"
#include "StringItem.h"
#include "CudaDevicePropItem.h"


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


// ( type -- )
class SizeofWord : public Word
{
public:
    SizeofWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        string type = AsString(interp->StackPop());
        int result = 1;
        if      (type == "FLOAT")    result = sizeof(float);
        else if (type == "INT")      result = sizeof(int);
        interp->StackPush(shared_ptr<IntItem>(new IntItem(result)));
    }
};


// ( address offset num type -- )
class PrintMemWord : public Word
{
public:
    PrintMemWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        string type = AsString(interp->StackPop());
        int num = AsInt(interp->StackPop());
        int offset = AsInt(interp->StackPop());
        auto address = interp->StackPop();

        if (type == "FLOAT")    printMemAsFloats(AsFloatStar(address), offset, num);
        else                    printMemAsInts(AsIntStar(address), offset, num);
    }

protected:
    void printMemAsFloats(float* addr, int offset, int num) {
        for (int i=0; i < num ; i++) {
            printf("%-8.4f ", addr[offset+i]);
        }
    }

    void printMemAsInts(int* addr, int offset, int num) {
        for (int i=0; i < num ; i++) {
            printf("%-8d  ", addr[offset+i]);
        }
    }
};


static void checkCudaCall(const cudaError_t res, const char* file, int line) {
    if (res != cudaSuccess) {
        stringstream builder;
        builder << cudaGetErrorString(res) << " " << file << ":" << line;
        throw builder.str();
    }
}

// ( index -- )
class CudaSetDeviceWord : public Word
{
public:
    CudaSetDeviceWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int index = AsInt(interp->StackPop());
        auto res = cudaSetDevice(index);
        checkCudaCall(res, __FILE__, __LINE__);
    }
};


// ( -- )
class CudaDeviceResetWord : public Word
{
public:
    CudaDeviceResetWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto res = cudaDeviceReset();
        checkCudaCall(res, __FILE__, __LINE__);
    }
};


// ( num-bytes -- addr )
class CudaMallocWord : public Word
{
public:
    CudaMallocWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int num_bytes = AsInt(interp->StackPop());

        void *result;
        auto res = cudaMalloc((void**)&result, num_bytes);
        checkCudaCall(res, __FILE__, __LINE__);
        interp->StackPush(AddressItem::New(result));
    }
};


// ( addr -- )
class CudaFreeWord : public Word
{
public:
    CudaFreeWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        void* addr = AsVoidStar(interp->StackPop());
        auto res = cudaFree(addr);
        checkCudaCall(res, __FILE__, __LINE__);
    }
};


// ( dst src num-bytes -- )
class CudaMemcpyHtDWord : public Word
{
public:
    CudaMemcpyHtDWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int num_bytes = AsInt(interp->StackPop());
        void* src = AsFloatStar(interp->StackPop());
        void* dst = AsFloatStar(interp->StackPop());

        auto res = cudaMemcpy(dst, src, num_bytes, cudaMemcpyHostToDevice);
        checkCudaCall(res, __FILE__, __LINE__);
    }
};


// ( dst src num-bytes -- )
class CudaMemcpyDtHWord : public Word
{
public:
    CudaMemcpyDtHWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int num_bytes = AsInt(interp->StackPop());
        void* src = AsFloatStar(interp->StackPop());
        void* dst = AsFloatStar(interp->StackPop());

        auto res = cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToHost);
        checkCudaCall(res, __FILE__, __LINE__);
    }
};


// ( devIndex -- cudaDeviceProp )
class CudaGetDevicePropertiesWord : public Word
{
public:
    CudaGetDevicePropertiesWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int devIndex = AsInt(interp->StackPop());

        cudaDeviceProp deviceProp;
        auto res = cudaGetDeviceProperties(&deviceProp, devIndex);
        checkCudaCall(res, __FILE__, __LINE__);
        interp->StackPush(CudaDevicePropItem::New(deviceProp));
    }
};


// ( cudaDeviceProp field -- value )
class DevPropWord : public Word
{
public:
    DevPropWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        string field = AsString(interp->StackPop());
        shared_ptr<StackItem> item = interp->StackPop();

        if (auto devPropItem = dynamic_cast<CudaDevicePropItem*>(item.get())) {
            const cudaDeviceProp& deviceProp = devPropItem->deviceProp();
            if (field == "name") {
                interp->StackPush(StringItem::New(string(deviceProp.name)));
            }
            else {
                throw string("Unknown dev prop field: ") + field;
            }
        }
        else {
            throw "Item was not a CudaDevicePropItem";
        }
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
    AddWord(shared_ptr<Word>(new SizeofWord("SIZEOF")));
    AddWord(shared_ptr<Word>(new PrintMemWord("PRINT-MEM")));
    AddWord(shared_ptr<Word>(new CudaSetDeviceWord("CUDA-SET-DEVICE")));
    AddWord(shared_ptr<Word>(new CudaDeviceResetWord("CUDA-DEVICE-RESET")));
    AddWord(shared_ptr<Word>(new CudaMallocWord("CUDA-MALLOC")));
    AddWord(shared_ptr<Word>(new CudaFreeWord("CUDA-FREE")));
    AddWord(shared_ptr<Word>(new CudaMemcpyHtDWord("CUDA-MEMCPY-HtD")));
    AddWord(shared_ptr<Word>(new CudaMemcpyDtHWord("CUDA-MEMCPY-DtH")));
    AddWord(shared_ptr<Word>(new CudaGetDevicePropertiesWord("CUDA-GET-DEVICE-PROPERTIES")));
    AddWord(shared_ptr<Word>(new DevPropWord("DEV-PROP")));
}

string CudaModule::ForthicCode() {
    string result(
    ": FLOAT   'FLOAT' ; "
    ": INT     'INT' ; "
    );
    return result;
}


// =============================================================================
// StackItem Converters


dim3 AsDim3(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetDim3*>(item.get())) {
        return i->GetDim3();
    }
    else {
        throw "Item does not implement IGetInt";
    }
}

float* AsFloatStar(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetAddress*>(item.get())) {
        return i->GetFloatStar();
    }
    else {
        throw item->StringRep() + ": does not implement IGetAddress";
    }
}

int* AsIntStar(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetAddress*>(item.get())) {
        return i->GetIntStar();
    }
    else {
        throw item->StringRep() + ": does not implement IGetAddress";
    }
}

void* AsVoidStar(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetAddress*>(item.get())) {
        return i->GetVoidStar();
    }
    else {
        throw item->StringRep() + ": does not implement IGetAddress";
    }
}
