#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <stack>
#include <vector>
#include <map>

#include "Module.h"

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


class IGetAddress {
public:
    virtual float* GetFloatStar() = 0;
    virtual int* GetIntStar() = 0;
    virtual void* GetVoidStar() = 0;
};

dim3 AsDim3(shared_ptr<StackItem> item);
float* AsFloatStar(shared_ptr<StackItem> item);
int* AsIntStar(shared_ptr<StackItem> item);
void* AsVoidStar(shared_ptr<StackItem> item);
