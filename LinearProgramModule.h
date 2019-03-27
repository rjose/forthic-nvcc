#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <stack>
#include <vector>
#include <map>

#include "Module.h"

using namespace std;


class LinearProgramModule : public Module
{
public:
    LinearProgramModule();

    virtual string ForthicCode();
};
