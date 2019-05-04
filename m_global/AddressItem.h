#pragma once
#include <string>

#include "../StackItem.h"
#include "I_AsFloatStar.h"
#include "I_AsIntStar.h"
#include "I_AsVoidStar.h"

using namespace std;


class AddressItem : public StackItem, public I_AsFloatStar, public I_AsIntStar, public I_AsVoidStar
{
public:
    AddressItem(void* address) : address(address) {};
    static shared_ptr<AddressItem> New(void* address);

    float* AsFloatStar();
    int* AsIntStar();
    void* AsVoidStar();

    virtual string StringRep();
    virtual string AsString();

protected:
    void* address;
};
