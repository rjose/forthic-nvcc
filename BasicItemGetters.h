#pragma once
#include <memory>
#include <string>
#include <vector>

#include "StackItem.h"

using namespace std;

class Module;


class IGetString {
public:
    virtual string GetString() = 0;
};

class IGetArray {
public:
    virtual vector<shared_ptr<StackItem>> GetArray() = 0;
};

class IGetValue {
public:
    virtual shared_ptr<StackItem> GetValue() = 0;
};

class IGetModule {
public:
    virtual shared_ptr<Module> GetModule() = 0;
};

string ForthicGetString(StackItem *item);
vector<shared_ptr<StackItem>> ForthicGetArray(StackItem *item);
shared_ptr<StackItem> ForthicGetValue(StackItem *item);
shared_ptr<Module> ForthicGetModule(StackItem *item);
