#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <stack>
#include <vector>
#include <map>

#include "../Module.h"

using namespace std;


class CudaModule : public Module
{
public:
    CudaModule();
    virtual string ForthicCode();

protected:
    // virtual shared_ptr<Word> treat_as_literal(string name);
};


class IGetDim3 {
public:
    virtual dim3 GetDim3() = 0;
};


dim3 AsDim3(shared_ptr<StackItem> item);
void checkCudaCall(const cudaError_t res, const char* file, int line);
