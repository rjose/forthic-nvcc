#pragma once
#include <memory>
#include <string>

#include "BasicConverters.h"
#include "StackItem.h"
#include "Module.h"

using namespace std;


class ModuleItem : public StackItem, public IAsModule
{
public:
	ModuleItem(shared_ptr<Module> mod) : mod(mod) {};
	virtual ~ModuleItem() {};
	shared_ptr<Module> AsModule();

protected:
	shared_ptr<Module> mod;
};
