#pragma once
#include "BasicConverters.h"
#include "StackItem.h"

using namespace std;

class VariableItem : public StackItem
{
public:
    VariableItem() : value(nullptr) {};
    virtual ~VariableItem() {};

    shared_ptr<StackItem> GetValue();
    void SetValue(shared_ptr<StackItem> new_value);

    virtual string StringRep();

protected:
    shared_ptr<StackItem> value;
};

