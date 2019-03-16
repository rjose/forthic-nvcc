#pragma once
#include <string>

#include "CudaModule.h"
#include "StackItem.h"

using namespace std;


class Dim3Item : public StackItem, public IGetDim3
{
public:
    Dim3Item(dim3 value) : value(value) {};
    dim3 GetDim3();

    virtual string StringRep();

protected:
    dim3 value;
};
