#pragma once
#include <memory>

#include "StackItem.h"

class IGetAddress {
public:
    virtual float* GetFloatStar() = 0;
    virtual int* GetIntStar() = 0;
    virtual void* GetVoidStar() = 0;
};

float* AsFloatStar(shared_ptr<StackItem> item);
int* AsIntStar(shared_ptr<StackItem> item);
void* AsVoidStar(shared_ptr<StackItem> item);
