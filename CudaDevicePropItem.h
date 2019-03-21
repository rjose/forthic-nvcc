#pragma once
#include <string>
#include <memory>
#include <cuda_runtime.h>

#include "StackItem.h"

using namespace std;

class CudaDevicePropItem : public StackItem
{
public:
    CudaDevicePropItem(cudaDeviceProp value) : value(value) {};
    static shared_ptr<CudaDevicePropItem> New(cudaDeviceProp value);

    const cudaDeviceProp& deviceProp();

    virtual string StringRep();
    virtual string AsString();

protected:
    cudaDeviceProp value;
};
