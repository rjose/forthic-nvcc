#pragma once
#include <string>

#include "../StackItem.h"

#include "I_AsInt.h"
#include "I_AsFloat.h"

using namespace std;


class IntItem : public StackItem, public I_AsInt, public I_AsFloat
{
public:
    IntItem(int value) : value(value) {};
    static shared_ptr<IntItem> New(int value);

    int AsInt();
    float AsFloat();

    virtual string StringRep();
    virtual string AsString();

protected:
    int value;
};
