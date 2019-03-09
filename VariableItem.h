#pragma once
#include "GlobalItemGetters.h"
#include "StackItem.h"

using namespace std;

class VariableItem : public StackItem, public IGetValue
{
public:
    VariableItem();
    virtual ~VariableItem();

    virtual shared_ptr<StackItem> GetValue();
    void SetValue(shared_ptr<StackItem> new_value);

protected:
    shared_ptr<StackItem> value;
};

