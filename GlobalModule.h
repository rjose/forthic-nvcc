#pragma once

#include <memory>
#include <string>
#include <stack>
#include <vector>
#include <map>

#include "Module.h"

using namespace std;


class GlobalModule : public Module
{
public:
    GlobalModule();
    virtual ~GlobalModule();

protected:
    virtual shared_ptr<Word> treat_as_literal(string name);

    shared_ptr<Word> treat_as_float(string name);
    shared_ptr<Word> treat_as_int(string name);
};

class IGetInt {
public:
    virtual int GetInt() = 0;
};

class IGetFloat {
public:
    virtual float GetFloat() = 0;
};

int   AsInt(shared_ptr<StackItem> item);
float AsFloat(shared_ptr<StackItem> item);
