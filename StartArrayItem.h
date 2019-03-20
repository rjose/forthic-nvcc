#pragma once
#include "StackItem.h"

using namespace std;

class StartArrayItem : public StackItem
{
public:
    StartArrayItem();
    virtual ~StartArrayItem();
    virtual string AsString();
};
