#pragma once
#include <string>

#include "StackItem.h"
#include "IGetAddress.h"

using namespace std;


class AddressItem : public StackItem, public IGetAddress
{
public:
    AddressItem(void* address) : address(address) {};
    static shared_ptr<AddressItem> New(void* address);

    float* GetFloatStar();
    int* GetIntStar();
    void* GetVoidStar();

    virtual string StringRep();
    virtual string AsString();

protected:
    void* address;
};
