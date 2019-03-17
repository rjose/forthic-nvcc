#pragma once
#include <string>

#include "GlobalModule.h"
#include "StackItem.h"

using namespace std;


class FloatItem : public StackItem, public IGetFloat, public IGetInt
{
public:
    FloatItem(float value);
    virtual ~FloatItem();

    float GetFloat();
    int GetInt();
    virtual string StringRep();

protected:
    float value;
};
