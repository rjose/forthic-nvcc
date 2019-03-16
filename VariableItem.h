#pragma once
#include "BasicConverters.h"
#include "StackItem.h"

using namespace std;

class VariableItem : public StackItem, public IAsVariable
{
public:
    VariableItem() : value(nullptr) {};
    virtual ~VariableItem() {};

    shared_ptr<VariableItem> AsVariable();
    shared_ptr<StackItem> GetValue();
    void SetValue(shared_ptr<StackItem> new_value);

protected:
    shared_ptr<StackItem> value;
};

