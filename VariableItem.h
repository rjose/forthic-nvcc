#pragma once
#include "./m_global/BasicConverters.h"
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
    virtual string AsString();

protected:
    shared_ptr<StackItem> value;
};

