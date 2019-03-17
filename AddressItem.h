#pragma once
#include <string>

#include "CudaModule.h"
#include "StackItem.h"

using namespace std;


class AddressItem : public StackItem, public IGetAddress
{
public:
    AddressItem(void* address) : address(address) {};

    float* GetFloatStar();
    int* GetIntStar();
    void* GetVoidStar();

    virtual string StringRep();

protected:
    void* address;
};
