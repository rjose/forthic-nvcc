#pragma once
#include <string>

#include "GlobalModule.h"
#include "../StackItem.h"

using namespace std;


class IntItem : public StackItem, public IGetInt, public IGetFloat
{
public:
    IntItem(int value) : value(value) {};
    static shared_ptr<IntItem> New(int value);

    int GetInt();
    float GetFloat();

    virtual string StringRep();
    virtual string AsString();

protected:
    int value;
};
