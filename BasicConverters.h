#pragma once
#include <memory>
#include <string>
#include <vector>

#include "StackItem.h"

using namespace std;

class Module;
class VariableItem;


class IAsString {
public:
    virtual string AsString() = 0;
};

class IAsArray {
public:
    virtual vector<shared_ptr<StackItem>> AsArray() = 0;
};

class IAsModule {
public:
    virtual shared_ptr<Module> AsModule() = 0;
};

string                        AsString(shared_ptr<StackItem> item);
vector<shared_ptr<StackItem>> AsArray(shared_ptr<StackItem> item);
shared_ptr<Module>            AsModule(shared_ptr<StackItem> item);
