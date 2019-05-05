#pragma once
#include <string>

#include "../StackItem.h"
#include "I_AsInt.h"
#include "I_AsFloat.h"

using namespace std;


class FloatItem : public StackItem, public I_AsFloat, public I_AsInt
{
public:
    FloatItem(float value);
    virtual ~FloatItem();

    float AsFloat();
    int AsInt();
    virtual string StringRep();
    virtual string AsString();

protected:
    float value;
};
