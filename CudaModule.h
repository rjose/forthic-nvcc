#pragma once

#include <memory>
#include <string>
#include <stack>
#include <vector>
#include <map>

#include "Module.h"

using namespace std;


class CudaModule : public Module
{
public:
    CudaModule();
    // virtual ~CudaModule();

protected:
    // virtual shared_ptr<Word> treat_as_literal(string name);
};

/*
class IGetFloat {
public:
    virtual float GetFloat() = 0;
};

float ForthicGetFloat(StackItem *item);
*/
