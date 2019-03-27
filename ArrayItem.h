#pragma once
#include <memory>
#include <string>
#include <vector>

#include "BasicConverters.h"
#include "StackItem.h"

using namespace std;


class ArrayItem : public StackItem, public IAsArray
{
public:
    ArrayItem(vector<shared_ptr<StackItem>> items) : items(items) {};
    vector<shared_ptr<StackItem>> AsArray();

    virtual string AsString();
    virtual string StringRep();

protected:
    vector<shared_ptr<StackItem>> items;
};
