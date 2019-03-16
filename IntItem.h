#pragma once
#include <string>

#include "GlobalModule.h"
#include "StackItem.h"

using namespace std;


class IntItem : public StackItem, public IGetInt, public IGetFloat
{
public:
    IntItem(int value) : value(value) {};

    int GetInt();
    float GetFloat();

    virtual string StringRep();

protected:
    int value;
};
